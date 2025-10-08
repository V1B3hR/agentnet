"""
Enterprise Integration Module for AgentNet

Phase 8 - Ecosystem & Integration implementation providing:
- Enterprise connectors (Slack, Teams, CRM systems)
- Developer platform components
- Cloud-native deployment utilities

This module enables AgentNet to integrate with enterprise systems
and provides tools for developers to build and deploy agent-based solutions.
"""

from .connectors import (
    SlackConnector,
    TeamsConnector,
    SalesforceConnector,
    HubSpotConnector,
    JiraConnector,
    ServiceNowConnector,
    Office365Connector,
    GoogleWorkspaceConnector,
    IntegrationConfig,
    Message,
    Document,
)

from .developer import (
    WorkflowDesigner,
    LowCodeInterface,
    AgentMarketplace,
    IDEExtension,
    WorkflowNode,
    WorkflowDefinition,
    AgentTemplate,
    MarketplacePlugin,
)

from .deployment import (
    KubernetesOperator,
    AutoScaler,
    MultiRegionDeployment,
    ServerlessAdapter,
    ClusterConfig,
    AutoScalingConfig,
    RegionConfig,
    ServerlessConfig,
)

__all__ = [
    # Enterprise Connectors
    "SlackConnector",
    "TeamsConnector",
    "SalesforceConnector",
    "HubSpotConnector",
    "JiraConnector",
    "ServiceNowConnector",
    "Office365Connector",
    "GoogleWorkspaceConnector",
    "IntegrationConfig",
    "Message",
    "Document",
    # Developer Platform
    "WorkflowDesigner",
    "LowCodeInterface",
    "AgentMarketplace",
    "IDEExtension",
    "WorkflowNode",
    "WorkflowDefinition",
    "AgentTemplate",
    "MarketplacePlugin",
    # Cloud-Native Deployment
    "KubernetesOperator",
    "AutoScaler",
    "MultiRegionDeployment",
    "ServerlessAdapter",
    "ClusterConfig",
    "AutoScalingConfig",
    "RegionConfig",
    "ServerlessConfig",
]
