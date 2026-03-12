#include <string>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>          // getenv

#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;



// libcurl write callback    
static size_t curl_write_cb(char* ptr, size_t size, size_t nmemb, std::string* out) {
    out->append(ptr, size * nmemb);    
    return size * nmemb; 
}


static std::string http_request(const std::string& url,const std::string& token, const std::string& post_body = "") {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("[QPU] curl_easy_init() failed ");

    struct curl_slist* headers = nullptr;
    std::string auth_hdr = "Authorization: Bearer " + token;
    headers = curl_slist_append(headers, auth_hdr.c_str());   
    headers = curl_slist_append(headers, "Content-Type: application/json");  

    std::string response_body;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);    
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);  
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,30L); 
     

    if (!post_body.empty()) {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_body.c_str());
    }  
 
    CURLcode rc = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) {
        throw std::runtime_error(std::string("[QPU] curl error: ")+ curl_easy_strerror(rc));  
    } 

    return response_body;
}

// subitmit QPU job - OpenQASM3 to IBM Quantum Sampler primitive 

std::string submit_qpu_job(const std::string& qasm,
                            const std::string&  /*backend_hint*/,
                            int num_shots) {
    const char* token_env = std::getenv("IBM_QUANTUM_TOKEN");
    const char* instance_env = std::getenv("IBM_QUANTUM_INSTANCE");
    const char* device_env= std::getenv("IBM_QUANTUM_BACKEND");  
           
    if (!token_env || !instance_env) {
        throw std::runtime_error("[QPU] IBM_QUANTUM_TOKEN and IBM_QUANTUM_INSTANCE must be set when backend_target='ibm_cloud' ");
    }

    const std::string token = token_env;    
    const std::string instance = instance_env;   
    const std::string device = device_env ? device_env : "ibm_brisbane";  

    json payload = {
        {"program_id","estimator"},
        {"backend", device},
        {"hub",instance},     
        {"params", {
            {"circuits", json::array({qasm})},
            {"observables",json::array()},      
            {"shots", num_shots}
        }}
    };

    const std::string url= "https://api.quantum-computing.ibm.com/runtime/jobs";
    const std::string body_str = payload.dump();   

    std::cout << "[QPU] Submitting job to " << device << " … " << std::endl;
    std::string response = http_request(url, token, body_str);

    json resp_json = json::parse(response);   
    if (resp_json.contains("error")) {
        throw std::runtime_error(
            "[QPU] Job submission error: " + resp_json["error"].dump());
    }    

    std::string job_id = resp_json["id"].get<std::string>();    
    std::cout << "[QPU] Job submitted - id=" << job_id << std::endl;
    return job_id;
}    


double poll_qpu_job(const std::string& job_id) {
    const char* token_env = std::getenv("IBM_QUANTUM_TOKEN");
    if (!token_env) throw std::runtime_error("[QPU] IBM_QUANTUM_TOKEN not set. ");
    const std::string token = token_env;
    const std::string url ="https://api.quantum-computing.ibm.com/runtime/jobs/" + job_id;

    // Back-off schedule - poll every 2 s, doubling up to 30 s, max is  20 minutes.
    int delay_s = 2; 
    int max_wait_s = 1200;
    int waited_s= 0;   
        
    while (waited_s < max_wait_s) {
        std::this_thread::sleep_for(std::chrono::seconds(delay_s));
        waited_s += delay_s;
        delay_s = std::min(delay_s * 2, 30);

        std::string response = http_request(url, token);
        json resp = json::parse(response);
        std::string status = resp.value("status", "UNKNOWN");

        std::cout << "[QPU] Job " << job_id << " - status=" << status << " (waited " << waited_s << "s) " << std::endl;

        if (status == "COMPLETED") {
            try {
                double ev = resp["results"][0]["data"]["evs"][0].get<double>();
                std::cout << "[QPU] Expectation value = " << ev << std::endl;
                return ev;
            } catch (const std::exception& ex) {
                throw std::runtime_error(std::string("[QPU] Failed to parse result: ") + ex.what()+ " \nRaw response: " + response);
            }
        }

        if (status == "ERROR" || status == "CANCELLED") {
            throw std::runtime_error(
                "[QPU] Job " + job_id + " ended with status=" + status + "  \nDetails: " + resp.dump(2));
        }
    }

    throw std::runtime_error(
        "[QPU] Timed out waiting for job " + job_id + " after " + std::to_string(max_wait_s) + "s. ");
}
