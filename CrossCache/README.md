# CrossCache

### Dependency Matrix

- vllm: v0.11.2
- vllm-ascend: 2b82320b
- torch_npu: 2.7.1
- CANN: 8.5.0
- HDK: 25.3.RC1

### How to build

##### Build the CrossCache server

```
git clone xxx --recurse-submodules
cd build;
```

- prepare dependency

```
bash build.sh prepare
```

- Prepare env

```
You should edit your env.local to configure your variable in correct way.
```

- build project

```
bash build.sh build
```

##### Build the client library

> Need NPU environment

```
cd python;
python3 setup.py build_ext --inplace
```

##### Run the crosscache

```
Now, we provide one demo call fake_client.py.

- run server
bin/crosscache

- run the client to test
python3 fake_client.py
```

### How to run with vllm

Change as following:

```python
# add to vllm/distributed/kv_transfer/kv_connector/factory.py
KVConnectorFactor.register_connector(
	"CrossCacheConnector",
	"vllm.distributed.kv_transfer.kv_connector.v1.crosscache.crosscache_connector",
	"CrossCacheConnector",
)
```

```sh
mkdir ${VLLM_SRC}/vllm/distributed/kv_transfer/kv_connector/v1/crosscache
cp -r python/utils ${VLLM_SRC}/vllm/distributed/kv_transfer/kv_connector/v1/crosscache
cp python/__init__.py ${VLLM_SRC}/vllm/distributed/kv_transfer/kv_connector/v1/crosscache
cp python/ipkey_utils.c* ${VLLM_SRC}/vllm/distributed/kv_transfer/kv_connector/v1/crosscache
cp python/adaptor.py ${VLLM_SRC}/vllm/distributed/kv_transfer/kv_connector/v1/crosscache
cp python/crosscache_connector.py ${VLLM_SRC}/vllm/distributed/kv_transfer/kv_connector/v1/crosscache
```

> ${VLLM_SRC} is the VLLM source directory which you built from.

##### run the vllm

```sh
export VLLM_LOGGING_LEVEL=DEBUG
vllm serve /workspace/models/qwen-2.5_7B_Instruct --no-enable-prefix-caching --gpu-memory-utilization 0.8 --kv-transfer-config '{"kv_connector":"CrossCacheConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"cache.server.host": "tcp://127.0.0.1", "cache.server.port": 5555}}'
vllm serve /workspace/models/qwen-2.5_7B_Instruct --no-enable-prefix-caching --gpu-memory-utilization 0.8 --kv-transfer-config '{"kv_connector":"CrossCacheConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"cache.server.host": "tcp://127.0.0.1", "cache.server.port": 5555}}' --host 0.0.0.0 --port 9000

```
