#!/usr/bin/env python3
"""
Test Suite for Phase 8 - Ecosystem & Integration Features

Tests the comprehensive Phase 8 enterprise ecosystem features:
- Enterprise Connectors: Slack, Teams, CRM, workflow, document processing
- Developer Platform: Visual designer, low-code interface, marketplace, IDE extensions
- Cloud-Native Deployment: Kubernetes, auto-scaling, multi-region, serverless
"""

import json
import tempfile
import unittest
import yaml
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

# Import Phase 8 modules for testing
from agentnet.enterprise import (
    # Connectors
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
    # Developer Platform
    WorkflowDesigner,
    LowCodeInterface,
    AgentMarketplace,
    IDEExtension,
    WorkflowNode,
    WorkflowDefinition,
    AgentTemplate,
    MarketplacePlugin,
    # Cloud-Native Deployment
    KubernetesOperator,
    AutoScaler,
    MultiRegionDeployment,
    ServerlessAdapter,
    ClusterConfig,
    AutoScalingConfig,
    RegionConfig,
    ServerlessConfig,
)


class TestEnterpriseConnectors(unittest.TestCase):
    """Test enterprise connector functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = IntegrationConfig(
            platform="test", api_endpoint="https://api.test.com", api_key="test_key"
        )

    def test_slack_connector(self):
        """Test Slack connector functionality."""
        connector = SlackConnector(self.config)

        # Test connection
        self.assertFalse(connector.is_connected)

        # Test async methods (simulate with sync for testing)
        import asyncio

        async def test_async():
            # Test connection
            connected = await connector.connect()
            self.assertTrue(connected)
            self.assertTrue(connector.is_connected)

            # Test health check
            healthy = await connector.health_check()
            self.assertTrue(healthy)

            # Test message sending
            sent = await connector.send_message("test-channel", "Hello World")
            self.assertTrue(sent)

            # Test channel creation
            channel_id = await connector.create_channel(
                "test-channel", "Test description"
            )
            self.assertTrue(channel_id.startswith("slack_channel_"))

            # Test disconnect
            disconnected = await connector.disconnect()
            self.assertTrue(disconnected)
            self.assertFalse(connector.is_connected)

        # Run async test
        asyncio.run(test_async())

    def test_teams_connector(self):
        """Test Microsoft Teams connector functionality."""
        connector = TeamsConnector(self.config)

        import asyncio

        async def test_async():
            connected = await connector.connect()
            self.assertTrue(connected)

            channel_id = await connector.create_channel(
                "team-channel", "Team description"
            )
            self.assertTrue(channel_id.startswith("teams_channel_"))

            await connector.disconnect()

        asyncio.run(test_async())

    def test_salesforce_connector(self):
        """Test Salesforce CRM connector functionality."""
        connector = SalesforceConnector(self.config)

        import asyncio

        async def test_async():
            connected = await connector.connect()
            self.assertTrue(connected)

            # Test contact operations
            contacts = await connector.get_contacts(limit=10)
            self.assertIsInstance(contacts, list)

            contact_id = await connector.create_contact(
                {
                    "first_name": "John",
                    "last_name": "Doe",
                    "email": "john.doe@example.com",
                }
            )
            self.assertTrue(contact_id.startswith("sf_contact_"))

            # Test opportunity operations
            opportunities = await connector.get_opportunities(limit=10)
            self.assertIsInstance(opportunities, list)

            opp_id = await connector.create_opportunity(
                {"name": "Test Opportunity", "amount": 10000, "stage": "Prospecting"}
            )
            self.assertTrue(opp_id.startswith("sf_opportunity_"))

            await connector.disconnect()

        asyncio.run(test_async())

    def test_jira_connector(self):
        """Test Jira workflow connector functionality."""
        connector = JiraConnector(self.config)

        import asyncio

        async def test_async():
            connected = await connector.connect()
            self.assertTrue(connected)

            # Test issue operations
            issues = await connector.get_issues("TEST", limit=10)
            self.assertIsInstance(issues, list)

            issue_id = await connector.create_issue(
                "TEST",
                {
                    "summary": "Test Issue",
                    "description": "Test Description",
                    "issue_type": "Bug",
                },
            )
            self.assertTrue(issue_id.startswith("jira_issue_"))

            updated = await connector.update_issue(issue_id, {"status": "In Progress"})
            self.assertTrue(updated)

            await connector.disconnect()

        asyncio.run(test_async())

    def test_office365_connector(self):
        """Test Office 365 document connector functionality."""
        connector = Office365Connector(self.config)

        import asyncio

        async def test_async():
            connected = await connector.connect()
            self.assertTrue(connected)

            # Test document operations
            documents = await connector.get_documents("Documents", limit=10)
            self.assertIsInstance(documents, list)

            test_doc = Document(
                id="test_doc",
                title="Test Document",
                content="Test content",
                format="docx",
                created_at=datetime.now(),
                author="test_user",
            )

            doc_id = await connector.create_document("Documents", test_doc)
            self.assertTrue(doc_id.startswith("o365_doc_"))

            search_results = await connector.search_documents("test", limit=5)
            self.assertIsInstance(search_results, list)

            await connector.disconnect()

        asyncio.run(test_async())


class TestDeveloperPlatform(unittest.TestCase):
    """Test developer platform functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_workflow_designer(self):
        """Test visual workflow designer functionality."""
        designer = WorkflowDesigner(storage_path=self.temp_dir)

        # Test workflow creation
        workflow = designer.create_workflow("Test Workflow", "Test Description")
        self.assertIsInstance(workflow, WorkflowDefinition)
        self.assertEqual(workflow.name, "Test Workflow")

        # Test node addition
        node_id = designer.add_node(
            workflow.id,
            "agent",
            "Test Agent",
            {"personality": {"logic": 0.8}},
            {"x": 100, "y": 200},
        )
        self.assertTrue(node_id.startswith("node_"))

        # Add another node
        node2_id = designer.add_node(
            workflow.id, "tool", "Test Tool", {"tool_type": "calculator"}
        )

        # Test node connection
        designer.connect_nodes(workflow.id, node_id, node2_id)

        # Test workflow validation
        validation = designer.validate_workflow(workflow.id)
        self.assertIsInstance(validation, dict)
        self.assertIn("valid", validation)

        # Test workflow export/import
        exported = designer.export_workflow(workflow.id)
        self.assertIsInstance(exported, dict)
        self.assertEqual(exported["name"], "Test Workflow")

        # Test workflow persistence
        designer.save_workflow(workflow.id)
        loaded_id = designer.load_workflow(workflow.id)
        self.assertEqual(loaded_id, workflow.id)

    def test_low_code_interface(self):
        """Test low-code/no-code interface functionality."""
        interface = LowCodeInterface(templates_path=self.temp_dir)

        # Test template listing
        templates = interface.list_templates()
        self.assertGreater(len(templates), 0)

        # Test specific template
        cs_template = interface.get_template("customer_service")
        self.assertIsNotNone(cs_template)
        self.assertEqual(cs_template.name, "Customer Service Agent")

        # Test parameter validation
        validation = interface.validate_parameters(
            "customer_service", {"company_name": "Acme Corp", "escalation_threshold": 3}
        )
        self.assertTrue(validation["valid"])

        # Test invalid parameters
        bad_validation = interface.validate_parameters(
            "customer_service", {"escalation_threshold": "invalid"}
        )
        self.assertFalse(bad_validation["valid"])

        # Test code generation
        code = interface.create_agent_from_template(
            "customer_service", {"company_name": "Acme Corp", "escalation_threshold": 5}
        )
        self.assertIsInstance(code, str)
        self.assertIn("CustomerServiceAgent", code)

    def test_agent_marketplace(self):
        """Test agent marketplace functionality."""
        marketplace = AgentMarketplace(storage_path=self.temp_dir)

        # Test plugin search
        all_plugins = marketplace.search_plugins()
        self.assertGreater(len(all_plugins), 0)

        # Test category filtering
        tools = marketplace.search_plugins(category="tools")
        self.assertGreater(len(tools), 0)

        # Test verified filtering
        verified = marketplace.search_plugins(verified_only=True)
        self.assertGreater(len(verified), 0)

        # Test search by query
        web_plugins = marketplace.search_plugins(query="web")
        self.assertGreater(len(web_plugins), 0)

        # Test popular plugins
        popular = marketplace.get_popular_plugins(limit=5)
        self.assertLessEqual(len(popular), 5)

        # Test top rated plugins
        top_rated = marketplace.get_top_rated_plugins(limit=3)
        self.assertLessEqual(len(top_rated), 3)

        # Test categories
        categories = marketplace.get_categories()
        self.assertIn("tools", categories)
        self.assertIn("integrations", categories)

        # Test plugin installation
        install_result = marketplace.install_plugin("web_scraper")
        self.assertTrue(install_result["success"])

        # Test plugin rating
        rated = marketplace.rate_plugin("web_scraper", 4.5)
        self.assertTrue(rated)

    def test_ide_extension(self):
        """Test IDE extension generation functionality."""
        ide_ext = IDEExtension()

        # Test supported IDEs
        supported = ide_ext.get_supported_ides()
        self.assertIn("vscode", supported)
        self.assertIn("jetbrains", supported)

        # Test VSCode extension generation
        vscode_files = ide_ext.generate_vscode_extension("AgentNet Dev Tools")
        self.assertIn("package.json", vscode_files)
        self.assertIn("src/extension.ts", vscode_files)
        self.assertIn("snippets/agentnet.json", vscode_files)

        # Validate package.json structure
        package_json = json.loads(vscode_files["package.json"])
        self.assertEqual(package_json["name"], "agentnet-agentnet-dev-tools")
        self.assertIn("activationEvents", package_json)

        # Test JetBrains plugin generation
        jetbrains_files = ide_ext.generate_jetbrains_plugin("AgentNet IDE")
        self.assertIn("META-INF/plugin.xml", jetbrains_files)
        self.assertIn("src/com/agentnet/CreateAgentAction.java", jetbrains_files)

        # Test unsupported IDE
        with self.assertRaises(ValueError):
            ide_ext.generate_extension_for_ide("unsupported", "test")


class TestCloudNativeDeployment(unittest.TestCase):
    """Test cloud-native deployment functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cluster_config = ClusterConfig(
            name="test-cluster", namespace="agentnet-test", replicas=2
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_kubernetes_operator(self):
        """Test Kubernetes operator functionality."""
        operator = KubernetesOperator(self.cluster_config)

        # Test individual resource generation
        namespace = operator.generate_namespace()
        self.assertEqual(namespace["metadata"]["name"], "agentnet-test")

        deployment = operator.generate_deployment()
        self.assertEqual(deployment["spec"]["replicas"], 2)

        service = operator.generate_service()
        self.assertEqual(service["spec"]["selector"]["app"], "agentnet")

        ingress = operator.generate_ingress()
        self.assertIn("rules", ingress["spec"])

        pvc = operator.generate_persistent_volume_claim()
        self.assertEqual(pvc["spec"]["resources"]["requests"]["storage"], "10Gi")

        configmap = operator.generate_configmap()
        self.assertIn("agentnet.yaml", configmap["data"])

        # Test all resources generation
        all_resources = operator.generate_all_resources()
        self.assertIn("namespace", all_resources)
        self.assertIn("deployment", all_resources)
        self.assertIn("service", all_resources)

        # Test manifest export
        operator.export_manifests(self.temp_dir)
        manifest_files = list(Path(self.temp_dir).glob("*.yaml"))
        self.assertGreater(len(manifest_files), 0)

    def test_auto_scaler(self):
        """Test auto-scaling functionality."""
        scaling_config = AutoScalingConfig(
            min_replicas=2, max_replicas=10, target_cpu_utilization=75
        )

        scaler = AutoScaler("test-cluster", "agentnet-test", scaling_config)

        # Test HPA generation
        hpa = scaler.generate_hpa()
        self.assertEqual(hpa["spec"]["minReplicas"], 2)
        self.assertEqual(hpa["spec"]["maxReplicas"], 10)

        # Test VPA generation
        vpa = scaler.generate_vpa()
        self.assertEqual(vpa["spec"]["targetRef"]["name"], "test-cluster-deployment")

        # Test PDB generation
        pdb = scaler.generate_pod_disruption_budget()
        self.assertEqual(pdb["spec"]["minAvailable"], 1)

        # Test scaling recommendations
        metrics = {"cpu_utilization": 85, "memory_utilization": 60}
        recommendations = scaler.get_scaling_recommendations(metrics)
        self.assertEqual(recommendations["scale_action"], "up")

        # Test scale down recommendation
        low_metrics = {"cpu_utilization": 20, "memory_utilization": 25}
        recommendations = scaler.get_scaling_recommendations(low_metrics)
        self.assertEqual(recommendations["scale_action"], "down")

    def test_multi_region_deployment(self):
        """Test multi-region deployment functionality."""
        deployment = MultiRegionDeployment("us-east-1")

        # Add regions
        us_east = RegionConfig(
            region="us-east-1",
            zones=["us-east-1a", "us-east-1b", "us-east-1c"],
            primary=True,
        )
        deployment.add_region(us_east)

        eu_west = RegionConfig(
            region="eu-west-1",
            zones=["eu-west-1a", "eu-west-1b"],
            compliance_tags=["gdpr"],
        )
        deployment.add_region(eu_west)

        # Test global load balancer generation
        global_lb = deployment.generate_global_load_balancer()
        self.assertEqual(len(global_lb["backends"]), 2)

        # Test region manifest generation
        us_manifest = deployment.generate_region_manifest("us-east-1")
        self.assertIn("deployment", us_manifest)

        # Test data locality policy
        policy = deployment.generate_data_locality_policy("eu-west-1")
        self.assertIn("compliance", yaml.safe_load(policy["data"]["policy.yaml"]))

        # Test deployment validation
        validation = deployment.validate_deployment()
        self.assertTrue(validation["valid"])

        # Test invalid configuration
        deployment.add_region(
            RegionConfig(
                region="ap-south-1",
                zones=["ap-south-1a"],
                primary=True,  # Second primary region
            )
        )
        validation = deployment.validate_deployment()
        self.assertFalse(validation["valid"])
        self.assertIn("Multiple primary regions", validation["issues"][0])

    def test_serverless_adapter(self):
        """Test serverless adapter functionality."""
        # Test AWS Lambda
        aws_config = ServerlessConfig(
            provider="aws",
            runtime="python3.9",
            timeout=300,
            memory=512,
            environment_variables={"AGENTNET_ENV": "production"},
            triggers=[{"type": "api_gateway", "path": "/api/agent"}],
        )

        aws_adapter = ServerlessAdapter(aws_config)

        # Test Lambda configuration generation
        lambda_config = aws_adapter.generate_aws_lambda("test-agent", "# Handler code")
        self.assertIn("Resources", lambda_config)
        self.assertIn("testagentFunction", lambda_config["Resources"])

        # Test handler template generation
        agent_code = "agent = AgentNet('Test', {'logic': 0.8})"
        handler = aws_adapter.generate_handler_template(agent_code)
        self.assertIn("lambda_handler", handler)
        self.assertIn("AgentNet", handler)

        # Test Azure Functions
        azure_config = ServerlessConfig(
            provider="azure", runtime="python3.9", timeout=300
        )

        azure_adapter = ServerlessAdapter(azure_config)

        azure_config_files = azure_adapter.generate_azure_function(
            "test-agent", "# Handler code"
        )
        self.assertIn("function.json", azure_config_files)
        self.assertIn("host.json", azure_config_files)
        self.assertIn("__init__.py", azure_config_files)

        # Test function deployment simulation
        deployment_info = aws_adapter.deploy_function("test-agent", b"dummy package")
        self.assertEqual(deployment_info["status"], "deployed")
        self.assertIn("endpoint", deployment_info)


class TestIntegratedP8Features(unittest.TestCase):
    """Test integrated Phase 8 features working together."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow from design to deployment."""
        # 1. Create agent using low-code interface
        interface = LowCodeInterface()
        agent_code = interface.create_agent_from_template(
            "customer_service", {"company_name": "Test Corp", "escalation_threshold": 3}
        )
        self.assertIn("CustomerServiceAgent", agent_code)

        # 2. Design workflow visually
        designer = WorkflowDesigner(storage_path=self.temp_dir)
        workflow = designer.create_workflow("Customer Support Workflow")

        # Add a start node to avoid orphaned node issues
        start_node = designer.add_node(workflow.id, "start", "Start", {})

        agent_node = designer.add_node(
            workflow.id, "agent", "Customer Service Agent", {"code": agent_code}
        )

        tool_node = designer.add_node(
            workflow.id, "tool", "CRM Connector", {"connector_type": "salesforce"}
        )

        # Connect nodes properly
        designer.connect_nodes(workflow.id, start_node, agent_node)
        designer.connect_nodes(workflow.id, agent_node, tool_node)

        # 3. Validate workflow
        validation = designer.validate_workflow(workflow.id)
        if not validation["valid"]:
            print(f"Validation errors: {validation['errors']}")
        self.assertTrue(validation["valid"])

        # 4. Generate deployment configuration
        cluster_config = ClusterConfig(name="customer-support", namespace="production")

        operator = KubernetesOperator(cluster_config)
        k8s_resources = operator.generate_all_resources()

        # 5. Add auto-scaling
        scaling_config = AutoScalingConfig(min_replicas=2, max_replicas=20)
        scaler = AutoScaler("customer-support", "production", scaling_config)
        hpa = scaler.generate_hpa()

        # 6. Generate serverless version
        serverless_config = ServerlessConfig(provider="aws", timeout=300, memory=1024)

        serverless_adapter = ServerlessAdapter(serverless_config)
        handler_code = serverless_adapter.generate_handler_template(agent_code)

        # Verify all components work together
        self.assertIsNotNone(agent_code)
        self.assertGreater(len(workflow.nodes), 0)
        self.assertIn("deployment", k8s_resources)
        self.assertIn("spec", hpa)
        self.assertIn("lambda_handler", handler_code)

    def test_marketplace_to_deployment_pipeline(self):
        """Test pipeline from marketplace plugin to deployed solution."""
        # 1. Browse marketplace
        marketplace = AgentMarketplace()
        web_plugins = marketplace.search_plugins(query="web", verified_only=True)
        self.assertGreater(len(web_plugins), 0)

        # 2. Install plugin
        plugin = web_plugins[0]
        install_result = marketplace.install_plugin(plugin.id)
        self.assertTrue(install_result["success"])

        # 3. Create workflow using plugin
        designer = WorkflowDesigner(storage_path=self.temp_dir)
        workflow = designer.create_workflow("Web Scraping Workflow")

        plugin_node = designer.add_node(
            workflow.id, "tool", plugin.name, {"plugin_id": plugin.id}
        )

        # 4. Deploy to multiple regions
        multi_region = MultiRegionDeployment("us-east-1")

        us_config = RegionConfig(
            region="us-east-1", zones=["us-east-1a", "us-east-1b"], primary=True
        )
        multi_region.add_region(us_config)

        eu_config = RegionConfig(
            region="eu-west-1",
            zones=["eu-west-1a", "eu-west-1b"],
            compliance_tags=["gdpr"],
        )
        multi_region.add_region(eu_config)

        # 5. Generate deployment manifests
        us_manifest = multi_region.generate_region_manifest("us-east-1")
        eu_manifest = multi_region.generate_region_manifest("eu-west-1")

        # Verify pipeline completion
        self.assertEqual(len(workflow.nodes), 1)
        self.assertIn("deployment", us_manifest)
        self.assertIn("deployment", eu_manifest)

        # Verify regional compliance
        us_deployment = us_manifest["deployment"]
        eu_deployment = eu_manifest["deployment"]

        self.assertEqual(us_deployment["metadata"]["labels"]["region"], "us-east-1")
        self.assertEqual(eu_deployment["metadata"]["labels"]["region"], "eu-west-1")


def run_p8_tests():
    """Run all Phase 8 enterprise ecosystem tests."""
    print("🧪 Running Phase 8 Enterprise Ecosystem Tests")
    print("=" * 60)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestEnterpriseConnectors,
        TestDeveloperPlatform,
        TestCloudNativeDeployment,
        TestIntegratedP8Features,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print results
    print("=" * 60)
    if result.wasSuccessful():
        print("✅ All Phase 8 tests passed successfully!")
        print(f"📊 Tests run: {result.testsRun}")
        print("🌐 Phase 8 Ecosystem & Integration features are ready!")
        print("\n🚀 Phase 8 Features Validated:")
        print("   ✓ Enterprise Connectors (Slack, Teams, CRM, Jira, O365)")
        print("   ✓ Developer Platform (Visual Designer, Low-Code, Marketplace)")
        print("   ✓ Cloud-Native Deployment (K8s, Auto-scaling, Multi-region)")
        print("   ✓ Serverless Adapters (AWS Lambda, Azure Functions)")
        print("   ✓ End-to-End Integration Workflows")
        return True
    else:
        print("❌ Some Phase 8 tests failed")
        print(f"📊 Tests run: {result.testsRun}")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"🚫 Errors: {len(result.errors)}")
        return False


if __name__ == "__main__":
    success = run_p8_tests()
    exit(0 if success else 1)
