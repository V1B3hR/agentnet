"""
Cloud-Native Deployment Components for AgentNet Phase 8

Provides infrastructure for cloud-native deployment:
- Kubernetes operator for AgentNet clusters
- Auto-scaling based on workload demand
- Multi-region deployment with data locality
- Serverless agent functions (AWS Lambda, Azure Functions)
"""

import json
import logging
import yaml
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Configuration for AgentNet cluster deployment."""

    name: str
    namespace: str = "agentnet"
    replicas: int = 3
    cpu_request: str = "500m"
    cpu_limit: str = "1000m"
    memory_request: str = "1Gi"
    memory_limit: str = "2Gi"
    storage_class: str = "standard"
    storage_size: str = "10Gi"
    ingress_enabled: bool = True
    tls_enabled: bool = True
    monitoring_enabled: bool = True


@dataclass
class AutoScalingConfig:
    """Configuration for auto-scaling policies."""

    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    scale_up_stabilization: int = 300  # seconds
    scale_down_stabilization: int = 300  # seconds
    custom_metrics: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RegionConfig:
    """Configuration for multi-region deployment."""

    region: str
    zones: List[str]
    primary: bool = False
    data_residency_requirements: List[str] = field(default_factory=list)
    compliance_tags: List[str] = field(default_factory=list)
    network_policies: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerlessConfig:
    """Configuration for serverless agent functions."""

    provider: str  # aws, azure, gcp
    runtime: str = "python3.9"
    timeout: int = 300
    memory: int = 512
    environment_variables: Dict[str, str] = field(default_factory=dict)
    triggers: List[Dict[str, Any]] = field(default_factory=list)


class KubernetesOperator:
    """Kubernetes operator for AgentNet clusters."""

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.resources: Dict[str, Dict[str, Any]] = {}

    def generate_namespace(self) -> Dict[str, Any]:
        """Generate namespace configuration."""
        namespace = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.config.namespace,
                "labels": {"app": "agentnet", "managed-by": "agentnet-operator"},
            },
        }
        self.resources["namespace"] = namespace
        return namespace

    def generate_deployment(self) -> Dict[str, Any]:
        """Generate deployment configuration."""
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{self.config.name}-deployment",
                "namespace": self.config.namespace,
                "labels": {"app": "agentnet", "component": "api-server"},
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {"app": "agentnet", "component": "api-server"}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": "agentnet", "component": "api-server"}
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "agentnet-api",
                                "image": "agentnet/api:latest",
                                "ports": [{"containerPort": 8000, "name": "http"}],
                                "resources": {
                                    "requests": {
                                        "cpu": self.config.cpu_request,
                                        "memory": self.config.memory_request,
                                    },
                                    "limits": {
                                        "cpu": self.config.cpu_limit,
                                        "memory": self.config.memory_limit,
                                    },
                                },
                                "env": [
                                    {
                                        "name": "AGENTNET_NAMESPACE",
                                        "value": self.config.namespace,
                                    },
                                    {
                                        "name": "AGENTNET_CLUSTER_NAME",
                                        "value": self.config.name,
                                    },
                                ],
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": "http"},
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/ready", "port": "http"},
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5,
                                },
                            }
                        ]
                    },
                },
            },
        }
        self.resources["deployment"] = deployment
        return deployment

    def generate_service(self) -> Dict[str, Any]:
        """Generate service configuration."""
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.config.name}-service",
                "namespace": self.config.namespace,
                "labels": {"app": "agentnet", "component": "api-server"},
            },
            "spec": {
                "selector": {"app": "agentnet", "component": "api-server"},
                "ports": [{"port": 80, "targetPort": "http", "name": "http"}],
                "type": "ClusterIP",
            },
        }
        self.resources["service"] = service
        return service

    def generate_ingress(self) -> Dict[str, Any]:
        """Generate ingress configuration."""
        if not self.config.ingress_enabled:
            return {}

        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{self.config.name}-ingress",
                "namespace": self.config.namespace,
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "cert-manager.io/cluster-issuer": (
                        "letsencrypt-prod" if self.config.tls_enabled else ""
                    ),
                },
            },
            "spec": {
                "rules": [
                    {
                        "host": f"{self.config.name}.example.com",
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": f"{self.config.name}-service",
                                            "port": {"number": 80},
                                        }
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
        }

        if self.config.tls_enabled:
            ingress["spec"]["tls"] = [
                {
                    "hosts": [f"{self.config.name}.example.com"],
                    "secretName": f"{self.config.name}-tls",
                }
            ]

        self.resources["ingress"] = ingress
        return ingress

    def generate_persistent_volume_claim(self) -> Dict[str, Any]:
        """Generate persistent volume claim for data storage."""
        pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{self.config.name}-data",
                "namespace": self.config.namespace,
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": self.config.storage_class,
                "resources": {"requests": {"storage": self.config.storage_size}},
            },
        }
        self.resources["pvc"] = pvc
        return pvc

    def generate_configmap(self) -> Dict[str, Any]:
        """Generate configuration map."""
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.config.name}-config",
                "namespace": self.config.namespace,
            },
            "data": {
                "agentnet.yaml": yaml.dump(
                    {
                        "cluster": {
                            "name": self.config.name,
                            "namespace": self.config.namespace,
                        },
                        "api": {"port": 8000, "workers": self.config.replicas},
                        "observability": {
                            "enabled": self.config.monitoring_enabled,
                            "metrics_port": 9090,
                        },
                    }
                )
            },
        }
        self.resources["configmap"] = configmap
        return configmap

    def generate_all_resources(self) -> Dict[str, Dict[str, Any]]:
        """Generate all Kubernetes resources."""
        self.generate_namespace()
        self.generate_deployment()
        self.generate_service()
        self.generate_ingress()
        self.generate_persistent_volume_claim()
        self.generate_configmap()

        return self.resources

    def export_manifests(self, output_dir: str):
        """Export all resources as YAML manifests."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for resource_type, resource in self.resources.items():
            if resource:  # Skip empty resources
                file_path = output_path / f"{resource_type}.yaml"
                with open(file_path, "w") as f:
                    yaml.dump(resource, f, default_flow_style=False)

                self.logger.info(f"Exported {resource_type} manifest to {file_path}")

    def deploy(self, kubectl_context: Optional[str] = None) -> bool:
        """Deploy resources to Kubernetes cluster (simulation)."""
        try:
            context_info = f" in context {kubectl_context}" if kubectl_context else ""
            self.logger.info(
                f"Deploying AgentNet cluster {self.config.name}{context_info}"
            )

            # In a real implementation, this would use kubectl or kubernetes client
            for resource_type, resource in self.resources.items():
                if resource:
                    self.logger.info(
                        f"Applying {resource_type}: {resource['metadata']['name']}"
                    )

            self.logger.info("AgentNet cluster deployed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to deploy cluster: {e}")
            return False


class AutoScaler:
    """Auto-scaling based on workload demand."""

    def __init__(self, cluster_name: str, namespace: str, config: AutoScalingConfig):
        self.cluster_name = cluster_name
        self.namespace = namespace
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_hpa(self) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler configuration."""
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.cluster_name}-hpa",
                "namespace": self.namespace,
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{self.cluster_name}-deployment",
                },
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.target_cpu_utilization,
                            },
                        },
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.target_memory_utilization,
                            },
                        },
                    },
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": self.config.scale_up_stabilization,
                        "policies": [
                            {"type": "Percent", "value": 100, "periodSeconds": 15}
                        ],
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": self.config.scale_down_stabilization,
                        "policies": [
                            {"type": "Percent", "value": 10, "periodSeconds": 60}
                        ],
                    },
                },
            },
        }

        # Add custom metrics if configured
        for metric in self.config.custom_metrics:
            hpa["spec"]["metrics"].append(metric)

        return hpa

    def generate_vpa(self) -> Dict[str, Any]:
        """Generate Vertical Pod Autoscaler configuration."""
        vpa = {
            "apiVersion": "autoscaling.k8s.io/v1",
            "kind": "VerticalPodAutoscaler",
            "metadata": {
                "name": f"{self.cluster_name}-vpa",
                "namespace": self.namespace,
            },
            "spec": {
                "targetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{self.cluster_name}-deployment",
                },
                "updatePolicy": {"updateMode": "Auto"},
                "resourcePolicy": {
                    "containerPolicies": [
                        {
                            "containerName": "agentnet-api",
                            "maxAllowed": {"cpu": "2", "memory": "4Gi"},
                            "minAllowed": {"cpu": "100m", "memory": "128Mi"},
                        }
                    ]
                },
            },
        }

        return vpa

    def generate_pod_disruption_budget(self) -> Dict[str, Any]:
        """Generate Pod Disruption Budget for high availability."""
        pdb = {
            "apiVersion": "policy/v1",
            "kind": "PodDisruptionBudget",
            "metadata": {
                "name": f"{self.cluster_name}-pdb",
                "namespace": self.namespace,
            },
            "spec": {
                "minAvailable": max(1, self.config.min_replicas // 2),
                "selector": {
                    "matchLabels": {"app": "agentnet", "component": "api-server"}
                },
            },
        }

        return pdb

    def get_scaling_recommendations(
        self, current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get scaling recommendations based on current metrics."""
        recommendations = {
            "scale_action": "none",
            "target_replicas": self.config.min_replicas,
            "reason": "No scaling needed",
        }

        cpu_usage = current_metrics.get("cpu_utilization", 0)
        memory_usage = current_metrics.get("memory_utilization", 0)

        if (
            cpu_usage > self.config.target_cpu_utilization
            or memory_usage > self.config.target_memory_utilization
        ):
            recommendations.update(
                {
                    "scale_action": "up",
                    "target_replicas": min(
                        self.config.max_replicas,
                        int(cpu_usage / self.config.target_cpu_utilization) + 1,
                    ),
                    "reason": f"High resource utilization: CPU {cpu_usage}%, Memory {memory_usage}%",
                }
            )
        elif (
            cpu_usage < self.config.target_cpu_utilization * 0.5
            and memory_usage < self.config.target_memory_utilization * 0.5
        ):
            recommendations.update(
                {
                    "scale_action": "down",
                    "target_replicas": max(
                        self.config.min_replicas,
                        int(cpu_usage / self.config.target_cpu_utilization),
                    ),
                    "reason": f"Low resource utilization: CPU {cpu_usage}%, Memory {memory_usage}%",
                }
            )

        return recommendations


class MultiRegionDeployment:
    """Multi-region deployment with data locality."""

    def __init__(self, primary_region: str):
        self.primary_region = primary_region
        self.regions: Dict[str, RegionConfig] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_region(self, config: RegionConfig):
        """Add a region to the deployment."""
        self.regions[config.region] = config
        self.logger.info(f"Added region: {config.region} (zones: {config.zones})")

    def generate_global_load_balancer(self) -> Dict[str, Any]:
        """Generate global load balancer configuration."""
        backends = []
        for region, config in self.regions.items():
            backends.append(
                {
                    "name": f"agentnet-{region}",
                    "description": f"AgentNet backend in {region}",
                    "groups": [
                        {
                            "group": f"https://www.googleapis.com/compute/v1/projects/PROJECT_ID/zones/{zone}/instanceGroups/agentnet-ig"
                            for zone in config.zones
                        }
                    ],
                    "healthChecks": [f"agentnet-health-check-{region}"],
                    "locality_lb_policy": "ROUND_ROBIN",
                }
            )

        global_lb = {
            "name": "agentnet-global-lb",
            "loadBalancingScheme": "EXTERNAL_MANAGED",
            "backends": backends,
            "routing": {
                "rules": [
                    {
                        "matchRules": [
                            {
                                "prefixMatch": "/api/",
                                "headerMatches": [
                                    {
                                        "headerName": "X-Data-Region",
                                        "exactMatch": region,
                                    }
                                ],
                            }
                        ],
                        "routeAction": {"backendService": f"agentnet-{region}"},
                    }
                    for region in self.regions.keys()
                ]
            },
        }

        return global_lb

    def generate_region_manifest(self, region: str) -> Dict[str, Any]:
        """Generate Kubernetes manifest for a specific region."""
        if region not in self.regions:
            raise ValueError(f"Region {region} not configured")

        config = self.regions[region]

        # Create cluster config with region-specific settings
        cluster_config = ClusterConfig(
            name=f"agentnet-{region}",
            namespace="agentnet",
            replicas=3 if config.primary else 2,
        )

        operator = KubernetesOperator(cluster_config)
        manifests = operator.generate_all_resources()

        # Add region-specific labels and annotations
        for resource in manifests.values():
            if "metadata" in resource:
                resource["metadata"].setdefault("labels", {}).update(
                    {"region": region, "primary": str(config.primary).lower()}
                )

                # Add compliance tags as annotations
                if config.compliance_tags:
                    resource["metadata"].setdefault("annotations", {}).update(
                        {
                            "compliance.agentnet.dev/tags": ",".join(
                                config.compliance_tags
                            )
                        }
                    )

        return manifests

    def generate_data_locality_policy(self, region: str) -> Dict[str, Any]:
        """Generate data locality policy for a region."""
        if region not in self.regions:
            raise ValueError(f"Region {region} not configured")

        config = self.regions[region]

        policy = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": f"data-locality-{region}", "namespace": "agentnet"},
            "data": {
                "policy.yaml": yaml.dump(
                    {
                        "region": region,
                        "data_residency": {
                            "requirements": config.data_residency_requirements,
                            "allowed_zones": config.zones,
                        },
                        "routing": {
                            "prefer_local": True,
                            "fallback_regions": [
                                r for r in self.regions.keys() if r != region
                            ],
                        },
                        "compliance": {
                            "tags": config.compliance_tags,
                            "audit_required": True,
                        },
                    }
                )
            },
        }

        return policy

    def validate_deployment(self) -> Dict[str, Any]:
        """Validate multi-region deployment configuration."""
        issues = []
        warnings = []

        # Check for primary region
        primary_regions = [r for r, c in self.regions.items() if c.primary]
        if len(primary_regions) == 0:
            issues.append("No primary region configured")
        elif len(primary_regions) > 1:
            issues.append(f"Multiple primary regions configured: {primary_regions}")

        # Check zone distribution
        for region, config in self.regions.items():
            if len(config.zones) < 2:
                warnings.append(
                    f"Region {region} has only {len(config.zones)} zones (recommended: 3+)"
                )

        # Check data residency conflicts
        all_requirements = set()
        for config in self.regions.values():
            for req in config.data_residency_requirements:
                if req in all_requirements:
                    warnings.append(f"Conflicting data residency requirement: {req}")
                all_requirements.add(req)

        return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings}


class ServerlessAdapter:
    """Serverless agent functions adapter."""

    def __init__(self, config: ServerlessConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_aws_lambda(
        self, function_name: str, handler_code: str
    ) -> Dict[str, Any]:
        """Generate AWS Lambda function configuration."""
        if self.config.provider != "aws":
            raise ValueError("Provider must be 'aws' for Lambda functions")

        # Convert function name to CloudFormation-compatible resource name
        resource_name = function_name.replace("-", "").replace("_", "")

        lambda_config = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"AgentNet serverless function: {function_name}",
            "Resources": {
                f"{resource_name}Function": {
                    "Type": "AWS::Lambda::Function",
                    "Properties": {
                        "FunctionName": function_name,
                        "Runtime": self.config.runtime,
                        "Handler": "lambda_function.lambda_handler",
                        "Code": {"ZipFile": handler_code},
                        "Timeout": self.config.timeout,
                        "MemorySize": self.config.memory,
                        "Environment": {"Variables": self.config.environment_variables},
                        "Role": {"Fn::GetAtt": [f"{resource_name}Role", "Arn"]},
                    },
                },
                f"{resource_name}Role": {
                    "Type": "AWS::IAM::Role",
                    "Properties": {
                        "AssumeRolePolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Principal": {"Service": "lambda.amazonaws.com"},
                                    "Action": "sts:AssumeRole",
                                }
                            ],
                        },
                        "ManagedPolicyArns": [
                            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
                        ],
                    },
                },
            },
        }

        # Add triggers
        for i, trigger in enumerate(self.config.triggers):
            if trigger["type"] == "api_gateway":
                lambda_config["Resources"][f"{resource_name}ApiGateway"] = {
                    "Type": "AWS::ApiGateway::RestApi",
                    "Properties": {
                        "Name": f"{function_name}-api",
                        "Description": f"API Gateway for {function_name}",
                    },
                }
            elif trigger["type"] == "s3":
                lambda_config["Resources"][f"{resource_name}S3Permission"] = {
                    "Type": "AWS::Lambda::Permission",
                    "Properties": {
                        "FunctionName": {"Ref": f"{resource_name}Function"},
                        "Action": "lambda:InvokeFunction",
                        "Principal": "s3.amazonaws.com",
                        "SourceArn": trigger["source_arn"],
                    },
                }

        return lambda_config

    def generate_azure_function(
        self, function_name: str, handler_code: str
    ) -> Dict[str, Any]:
        """Generate Azure Function configuration."""
        if self.config.provider != "azure":
            raise ValueError("Provider must be 'azure' for Azure Functions")

        function_json = {
            "bindings": [
                {
                    "authLevel": "function",
                    "type": "httpTrigger",
                    "direction": "in",
                    "name": "req",
                    "methods": ["get", "post"],
                },
                {"type": "http", "direction": "out", "name": "$return"},
            ]
        }

        host_json = {
            "version": "2.0",
            "logging": {
                "applicationInsights": {"samplingSettings": {"isEnabled": True}}
            },
            "functionTimeout": f"00:{self.config.timeout // 60:02d}:{self.config.timeout % 60:02d}",
        }

        requirements_txt = """
azure-functions
agentnet
"""

        return {
            "function.json": json.dumps(function_json, indent=2),
            "host.json": json.dumps(host_json, indent=2),
            "__init__.py": handler_code,
            "requirements.txt": requirements_txt,
        }

    def generate_handler_template(self, agent_code: str) -> str:
        """Generate serverless handler template."""
        if self.config.provider == "aws":
            template = f"""
import json
import logging
from agentnet import AgentNet

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize agent
{agent_code}

def lambda_handler(event, context):
    try:
        # Extract input from event
        if 'body' in event:
            body = json.loads(event['body'])
            prompt = body.get('prompt', '')
        else:
            prompt = event.get('prompt', '')
        
        # Process with agent
        result = agent.generate_reasoning_tree(prompt)
        
        return {{
            'statusCode': 200,
            'headers': {{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }},
            'body': json.dumps(result)
        }}
    except Exception as e:
        logger.error(f"Error processing request: {{e}}")
        return {{
            'statusCode': 500,
            'body': json.dumps({{'error': str(e)}})
        }}
"""
        elif self.config.provider == "azure":
            template = f"""
import logging
import json
import azure.functions as func
from agentnet import AgentNet

# Initialize agent
{agent_code}

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('AgentNet function processed a request.')
    
    try:
        # Get prompt from request
        prompt = req.params.get('prompt')
        if not prompt:
            try:
                req_body = req.get_json()
                prompt = req_body.get('prompt')
            except ValueError:
                pass
        
        if not prompt:
            return func.HttpResponse("Please provide a prompt parameter", status_code=400)
        
        # Process with agent
        result = agent.generate_reasoning_tree(prompt)
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            headers={{"Content-Type": "application/json"}}
        )
    except Exception as e:
        logging.error(f"Error processing request: {{e}}")
        return func.HttpResponse(
            json.dumps({{"error": str(e)}}),
            status_code=500
        )
"""
        else:
            template = f"""
# Generic serverless handler template
# Provider: {self.config.provider}

{agent_code}

def handler(event, context):
    # Implement handler logic for {self.config.provider}
    pass
"""

        return template

    def deploy_function(
        self, function_name: str, code_package: bytes
    ) -> Dict[str, Any]:
        """Deploy serverless function (simulation)."""
        try:
            self.logger.info(
                f"Deploying {self.config.provider} function: {function_name}"
            )

            # Simulate deployment process
            endpoint = f"https://{function_name}.{self.config.provider}.example.com"

            deployment_info = {
                "function_name": function_name,
                "provider": self.config.provider,
                "runtime": self.config.runtime,
                "endpoint": endpoint,
                "memory": self.config.memory,
                "timeout": self.config.timeout,
                "deployed_at": datetime.now().isoformat(),
                "status": "deployed",
            }

            self.logger.info(f"Function deployed successfully: {endpoint}")
            return deployment_info
        except Exception as e:
            self.logger.error(f"Failed to deploy function: {e}")
            return {"status": "failed", "error": str(e)}
