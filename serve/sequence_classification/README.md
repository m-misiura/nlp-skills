## Deploy classifier on RHOAI

### Prerequisites

RHOAI cluster with the following operators:

__GPU__ -- follow [this guide](https://docs.nvidia.com/datacenter/cloud-native/openshift/latest/steps-overview.html) and install:
- Node Feature Discovery Operator (4.17.0-202505061137 provided by Red Hat):
    - ensure to create an instance of NodeFeatureDiscovery using the NodeFeatureDiscovery tab
- NVIDIA GPU Operator (25.3.0 provided by NVIDIA Corporation)
    - ensure to create an instance of ClusterPolicy using the ClusterPolicy tab

__Model Serving__: 
- Red Hat OpenShift Service Mesh 2 (2.6.7-0 provided by Red Hat, Inc.)
- Red Hat OpenShift Serverless (1.35.1 provided by Red Hat)
__Authentication__: 
- Red Hat - Authorino Operator (1.2.1 provided by Red Hat)

__AI Platform__:
- Red Hat OpenShift AI (2.20.0 provided by Red Hat, Inc.):
    - in the `DataScienceInitialization` resource, set the value of `managementState` for the `serviceMesh` component to `Removed`
    - in the `default-dsc`, ensure:
        1. `trustyai` `managementState` is set to `Managed`
        2. `kserve` is set to:
            ```yaml
            kserve:
                defaultDeploymentMode: RawDeployment
                managementState: Managed
                nim:
                    managementState: Managed
                rawDeploymentServiceConfig: Headless
                serving:
                    ingressGateway:
                    certificate:
                        type: OpenshiftDefaultIngress
                    managementState: Removed
                    name: knative-serving
            ```


### Deployin the model

1. Create a new project, using e.g. openshift cli:
```bash
oc new-project classify
```

2. Deploy all the required compontents:
```bash
oc apply -f serve/sequence_classification/bert-tiny-deployment.yaml
```

3. Expose the service:
```bash
oc expose service bert-text-classifier-predictor --name=llm-router
```

4. Get the model route:
```bash
export ROUTER_ROUTE="http://$(oc get route llm-router -o jsonpath='{.spec.host}')"
```

5. Test the model with a sample request:
```bash
curl -X POST "${ROUTER_ROUTE}/api/v1/task/text-classification" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "How does the structure and function of plasmodesmata affect cell-to-cell communication and signaling in plant tissues, particularly in response to environmental stresses?",
    "model_id": "bert-tiny-llm-router"
  }'
``` 