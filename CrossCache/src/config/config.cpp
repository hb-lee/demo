#include "conf.h"
#include "log.h"

#include <json/json.h>
#include <unistd.h>
#include <math.h>
#include <fstream>
#include <new>

typedef std::map<std::string, std::string> map_t;

static bool fileToJson(const std::string& filepath, Json::Value& dst)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        log_error("input file stream is not open");
        return false;
    }

    Json::CharReaderBuilder reader;
    std::string errs;

    bool ok = Json::parseFromStream(reader, file, &dst, &errs);
    if (!ok)
        log_error("parse file to json failed, filepath:%s", filepath.c_str());
    
    file.close();
    return ok;
}

static int parseBasic(Json::Value &basic, map_t &mem)
{
    if (!basic["Port"].isUInt()) {
        log_error("Basic/Port must be 'Uint32'");
        return -1;
    }

    if (!basic["ChunkSize"].isUInt()) {
        log_error("Basic/ChunkSize must be 'Uint32'");
        return -1;
    }

    if (!basic["CacheDir"].isString()) {
        log_error("Basic/CacheDir must be 'String'");
        return -1;
    }

    uint32_t port = basic["Port"].asUInt();
    mem.insert(std::pair<std::string, std::string>(BASIC_PORT, std::to_string(port)));
    uint32_t chunksize = basic["ChunkSize"].asUInt();
    mem.insert(std::pair<std::string, std::string>(BASIC_CHUNKSZ, std::to_string(chunksize)));
    std::string cachedir = basic["CacheDir"].asString();
    mem.insert(std::pair<std::string, std::string>(BASIC_CACHEDIR, cachedir));
    return 0;
}

static int parseRequest(Json::Value &req, map_t &mem)
{
    if (!req["LoadWorker"].isUInt()) {
        log_error("Request/LoadWorker must be 'Uint32'");
        return -1;
    }
    if (!req["StoreWorker"].isUInt()) {
        log_error("Request/StoreWorker must be 'Uint32'");
        return -1;
    }
    if (!req["CommWorker"].isUInt()) {
        log_error("Request/CommWorker must be 'Uint32'");
        return -1;
    }

    uint32_t lworker = req["LoadWorker"].asUInt();
    mem.insert(std::pair<std::string, std::string>(REQ_LWORKER, std::to_string(lworker)));
    uint32_t sworker = req["StoreWorker"].asUInt();
    mem.insert(std::pair<std::string, std::string>(REQ_SWORKER, std::to_string(sworker)));
    uint32_t cworker = req["CommWorker"].asUInt();
    mem.insert(std::pair<std::string, std::string>(REQ_CWORKER, std::to_string(cworker)));
    return 0;
}

class Config
{
    private:
        static map_t memDB;

    public:
        explicit Config() {}
        ~Config() {}

        static int Prepare(const char *file)
        {
            Json::Value root;
            std::string path(file);

            if (!fileToJson(path, root)) 
                return -1;

            if (parseBasic(root["Basic"], memDB) != 0)
                return -1;

                return parseRequest(root["Request"], memDB);
        }

        static uint32_t GetU32(const char *key)
        {
            map_t::iterator iter = memDB.find(key);
            if (iter == memDB.end())
                return 0;

            return (uint32_t)std::stoul(it->second);
        }
        
        static uint64_t GetU64(const char *key)
        {
            map_t::iterator iter = memDB.find(key);
            if (iter == memDB.end())
                return 0;

            return (uint64_t)std::stoul(it->second);
        }

        static const char *GetString(const char *key)
        {
            map_t::iterator iter = memDB.find(key);
            if (iter == memDB.end())
                return NULL;

            return it->second.c_str();
        }
};

mem_t Config::memDB;

int config_init(const char *file)
{
    return Config::Prepare(file);
}

void config_exit()
{
}

uint32_t config_get_u32(const char *key)
{
    return Config::GetU32(key);
}

uint64_t config_get_u64(const char *key)
{
    return Config::GetU64(key);
}

const char *config_get_string(const char *key)
{
    return Config::GetString(key);
}