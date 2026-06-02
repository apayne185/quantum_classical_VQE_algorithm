#include <string>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>         

#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;



static size_t curl_write_cb(char* ptr, size_t size, size_t nmemb, std::string* out) {
    out->append(ptr, size * nmemb);    
    return size * nmemb; 
}


static std::string http_request(const std::string& url,const std::string& bearer_token,  const std::string& crn, const std::string& post_body = "") {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("[QPU] curl_easy_init() failed ");

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("Authorization: Bearer " + bearer_token).c_str());
    headers = curl_slist_append(headers, ("Service-CRN: " + crn).c_str());
    headers = curl_slist_append(headers, "IBM-API-Version: 2025-05-01");
    headers = curl_slist_append(headers,"Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: application/json");

    std::string response_body;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);    
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);  
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,30L); 
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
     
    if (!post_body.empty()) {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_body.c_str());
    }  
     
    // Retry up to 3 times on transient network errors (timeout, connection reset, etc.)
    CURLcode rc = CURLE_OK;
    for (int attempt = 0; attempt < 3; ++attempt) {
        response_body.clear();
        rc = curl_easy_perform(curl);
        if (rc == CURLE_OK) break;
        const bool transient = (rc == CURLE_OPERATION_TIMEDOUT
                                || rc == CURLE_COULDNT_CONNECT
                                || rc == CURLE_RECV_ERROR
                                || rc == CURLE_SEND_ERROR);
        if (!transient || attempt == 2) break;
        std::cerr << "[QPU] curl transient error (" << curl_easy_strerror(rc)
                  << "), retrying in 2s..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK)
        throw std::runtime_error(std::string("[QPU] curl error: ") + curl_easy_strerror(rc));

    return response_body;
}




static std::string get_iam_bearer_token(const std::string& api_key) {
    CURL* curl = curl_easy_init();   
    if (!curl) throw std::runtime_error("[QPU] curl_easy_init() failed (IAM)  ");

    const std::string iam_url = "https://iam.cloud.ibm.com/identity/token";
    const std::string post_fields = "grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey"
        "&apikey=" + api_key;    
    

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");
    headers = curl_slist_append(headers,"Accept: application/json");
    
    std::string response_body;
    curl_easy_setopt(curl, CURLOPT_URL, iam_url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,post_fields.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,15L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    
    CURLcode rc = curl_easy_perform(curl);   
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);   
    
    if (rc != CURLE_OK)
        throw std::runtime_error(std::string("[QPU] IAM token exchange failed:  ") + curl_easy_strerror(rc));

    json resp = json::parse(response_body);
    if (!resp.contains("access_token"))
        throw std::runtime_error("[QPU] IAM token exchange failed: access_token not present in response");

    return resp["access_token"].get<std::string>();
}







// Submit OpenQASM3 circuit to IBM Quantum EstimatorV2 via REST
std::string submit_qpu_job(const std::string& qasm, const std::string& /*backend_hint*/, int num_shots) {
    const char* api_key_env = std::getenv("IBM_QUANTUM_TOKEN");
    const char* crn_env = std::getenv("IBM_QUANTUM_INSTANCE");
    const char* device_env= std::getenv("IBM_QUANTUM_BACKEND");  
    const char* region_env= std::getenv("IBM_QUANTUM_REGION");
           
    if (!api_key_env || !crn_env) {
        throw std::runtime_error("[QPU] IBM_QUANTUM_TOKEN and IBM_QUANTUM_INSTANCE must be set when backend_target='ibm_cloud' ");
    }

    const std::string token = api_key_env;    
    const std::string instance = crn_env;   
    const std::string device = device_env ? device_env : "ibm_brisbane";  
    const std::string region= region_env ? region_env : "us-east";  

    std::cout << "[QPU] Exchanging API key for IAM bearer token..." << std::endl;
    const std::string bearer = get_iam_bearer_token(token);
    const std::string crn = instance;

    json observable = "Z";          // minimal observable for connectivity validation

    json payload = {
        {"program_id","estimator"},
        {"backend", device},
        {"instance",instance},     
        {"params", {
            {"pubs",json::array({ json::array({qasm, observable}) })},
            {"version", 2},
            {"options", {
                {"default_shots",num_shots},
                {"resilience_level", 1},
                {"optimization_level",1}
            }}
        }}
    };


    std::string base_url= "https://quantum.cloud.ibm.com/api/v1"; 
    if (region == "eu-de")
        base_url = "https://eu-de.quantum.cloud.ibm.com/api/v1";
    const std::string jobs_url = base_url + "/jobs";
    const std::string body_str = payload.dump();
     

    std::cout << "[QPU] Submitting job to " << device << " (" << jobs_url << ") ..." << std::endl;
    std::string response = http_request(jobs_url, bearer, crn, body_str);
    json resp_json = json::parse(response);

    if (resp_json.contains("errors")) {
        throw std::runtime_error(
            "[QPU] Job submission error:  " + resp_json["errors"].dump());
    }    
    if (!resp_json.contains("id")) {
        throw std::runtime_error(
            "[QPU] Unexpected response (no id field): " + response);
    }


    std::string job_id = resp_json["id"].get<std::string>();    
    std::cout << "[QPU] Job submitted - id=" << job_id << std::endl;
    return job_id;
}       






double poll_qpu_job(const std::string& job_id) {
    const char* api_key_env = std::getenv("IBM_QUANTUM_TOKEN");
    const char* crn_env= std::getenv("IBM_QUANTUM_INSTANCE");
    const char* region_env = std::getenv("IBM_QUANTUM_REGION");

    if (!api_key_env) throw std::runtime_error("[QPU] IBM_QUANTUM_TOKEN not set. ");  

    const std::string bearer = get_iam_bearer_token(api_key_env);
    const std::string crn= crn_env ? crn_env : "";
    const std::string region = region_env ? region_env : "us-east";

    std::string base_url = "https://quantum.cloud.ibm.com/api/v1";
    if (region == "eu-de")
        base_url = "https://eu-de.quantum.cloud.ibm.com/api/v1";   

    const std::string status_url = base_url + "/jobs/" + job_id;
    const std::string result_url = base_url + "/jobs/" + job_id + "/results"; 


    int delay_s = 2;                // exponential backoff: 2s -> 30s cap
    int max_wait_s = 1200;          // 20 min timeout
    int waited_s= 0;   
        
    while (waited_s < max_wait_s) {
        std::this_thread::sleep_for(std::chrono::seconds(delay_s));
        waited_s += delay_s;
        delay_s = std::min(delay_s * 2, 30);

        std::string status_resp = http_request(status_url, bearer, crn);
        json status_json = json::parse(status_resp);
        std::string status = status_json.value("status","UNKNOWN");

        std::cout << "[QPU] Job " << job_id << " - status=" << status << " (waited " << waited_s << "s) " << std::endl;

        if (status == "COMPLETED" || status == "Completed") {
            std::string result_resp = http_request(result_url, bearer, crn);
            json result_json = json::parse(result_resp);

            try {
                if (!result_json.contains("results") || !result_json["results"].is_array()
                        || result_json["results"].empty()) {
                    throw std::runtime_error("results array is missing or empty");
                }
                auto& evs = result_json["results"][0]["data"]["evs"];
                double ev = 0.0;
                if (evs.is_array()) {
                    if (evs.empty()) throw std::runtime_error("evs array is empty");
                    ev = evs[0].get<double>();
                } else {
                    ev = evs.get<double>();
                }
                std::cout << "[QPU] Expectation value = " << ev << std::endl;
                return ev;

            } catch (const std::exception& ex) {
                throw std::runtime_error(std::string("[QPU] Failed to parse result: ") + ex.what());
            }
        }
        
        if (status == "ERROR" || status == "CANCELLED" || status == "Cancelled") {
            // Omit raw JSON dump — status_json may echo back request headers containing tokens.
            std::string reason = status_json.value("error", status_json.value("message", "no details available"));
            throw std::runtime_error(
                "[QPU] Job " + job_id + " ended with status=" + status + ". Reason: " + reason);
        }
    }

    throw std::runtime_error(
        "[QPU] Timed out waiting for job " + job_id + " after " + std::to_string(max_wait_s) + "s. ");
}   

