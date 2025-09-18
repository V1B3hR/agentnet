#!/usr/bin/env python3
"""
Test Suite for P6 Enterprise Hardening Features

Tests the comprehensive P6 enterprise features:
- Export Controls: Data classification and content redaction
- Audit Workflow: Comprehensive audit logging and compliance reporting
- Plugin SDK: Extensible plugin framework with security controls
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from agentnet.audit import (
    AuditDashboard,
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditQuery,
    AuditSeverity,
    AuditStorage,
    AuditWorkflow,
)

# Import P6 modules for testing
from agentnet.compliance import (
    ClassificationLevel,
    CompliancePolicy,
    ComplianceReporter,
    ContentRedactor,
    DataClassifier,
    ExportControlManager,
)
from agentnet.plugins import (
    Plugin,
    PluginInfo,
    PluginLoader,
    PluginManager,
    PluginSandbox,
    PluginStatus,
    SecurityPolicy,
)


class TestExportControls(unittest.TestCase):
    """Test export control functionality."""

    def setUp(self):
        self.export_manager = ExportControlManager()

    def test_data_classification(self):
        """Test data classification functionality."""
        classifier = DataClassifier()

        # Test public content
        level, detections = classifier.classify_content("This is public information.")
        self.assertEqual(level, ClassificationLevel.PUBLIC)
        self.assertEqual(len(detections), 0)

        # Test PII content
        level, detections = classifier.classify_content("SSN: 123-45-6789")
        self.assertEqual(level, ClassificationLevel.RESTRICTED)
        self.assertGreater(len(detections), 0)

    def test_content_redaction(self):
        """Test content redaction functionality."""
        redactor = ContentRedactor()

        content = "User SSN: 123-45-6789 and email: user@example.com"
        redacted, redactions = redactor.redact_content(
            content, ClassificationLevel.RESTRICTED
        )

        self.assertIn("[SSN-REDACTED]", redacted)
        self.assertNotIn("123-45-6789", redacted)
        self.assertGreater(len(redactions), 0)

    def test_export_evaluation(self):
        """Test export eligibility evaluation."""
        evaluation = self.export_manager.evaluate_export_eligibility(
            "This contains a SSN: 123-45-6789", "external_partner"
        )

        self.assertIn("evaluation", evaluation)
        self.assertIn("export_decision", evaluation)
        self.assertEqual(evaluation["evaluation"]["classification_level"], "restricted")

    def test_compliance_reporting(self):
        """Test export control compliance reporting."""
        # Generate some evaluations
        test_content = ["Public information", "SSN: 123-45-6789", "CONFIDENTIAL data"]

        for content in test_content:
            self.export_manager.evaluate_export_eligibility(content, "external")

        audit_trail = self.export_manager.get_export_audit_trail()
        self.assertEqual(len(audit_trail), 3)


class TestAuditWorkflow(unittest.TestCase):
    """Test audit workflow functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.audit_storage = AuditStorage(os.path.join(self.temp_dir, "test_audit.db"))
        self.audit_logger = AuditLogger(self.audit_storage)
        self.audit_workflow = AuditWorkflow(self.audit_logger)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_audit_event_creation(self):
        """Test audit event creation and logging."""
        event = AuditEvent(
            event_type=AuditEventType.AGENT_INFERENCE,
            severity=AuditSeverity.LOW,
            user_id="test_user",
            agent_id="test_agent",
            action="test_action",
            details={"test": "data"},
        )

        self.audit_logger.log_event(event)

        # Verify event was stored
        events = self.audit_storage.get_events(limit=1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].user_id, "test_user")

    def test_audit_queries(self):
        """Test audit event querying."""
        # Create test events
        events = [
            AuditEvent(event_type=AuditEventType.USER_LOGIN, user_id="user1"),
            AuditEvent(
                event_type=AuditEventType.SECURITY_ALERT, severity=AuditSeverity.HIGH
            ),
            AuditEvent(event_type=AuditEventType.POLICY_VIOLATION, agent_id="agent1"),
        ]

        for event in events:
            self.audit_logger.log_event(event)

        # Query by severity
        high_severity_query = AuditQuery(severity_levels=[AuditSeverity.HIGH])
        high_events = self.audit_storage.get_events(high_severity_query)
        self.assertEqual(len(high_events), 1)

        # Query by event type
        login_query = AuditQuery(event_types=[AuditEventType.USER_LOGIN])
        login_events = self.audit_storage.get_events(login_query)
        self.assertEqual(len(login_events), 1)

    def test_compliance_dashboard(self):
        """Test compliance dashboard generation."""
        # Create sample events
        events = [
            AuditEvent(
                event_type=AuditEventType.AGENT_INFERENCE, severity=AuditSeverity.LOW
            ),
            AuditEvent(
                event_type=AuditEventType.SECURITY_ALERT, severity=AuditSeverity.HIGH
            ),
            AuditEvent(
                event_type=AuditEventType.POLICY_VIOLATION,
                severity=AuditSeverity.MEDIUM,
            ),
        ]

        for event in events:
            self.audit_logger.log_event(event)

        dashboard = AuditDashboard(self.audit_storage)
        dashboard_data = dashboard.generate_compliance_dashboard(days=1)

        self.assertIn("summary", dashboard_data)
        self.assertIn("total_events", dashboard_data["summary"])
        self.assertEqual(dashboard_data["summary"]["total_events"], 3)

    def test_soc2_reporting(self):
        """Test SOC2 compliance reporting."""
        compliance_reporter = ComplianceReporter(self.temp_dir)

        # Record some audit events
        compliance_reporter.record_audit_event(
            "data_access", {"user_id": "test_user", "resource": "customer_data"}
        )

        # Generate SOC2 report
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        report = compliance_reporter.generate_soc2_report(start_date, end_date)

        self.assertEqual(report.report_type, "SOC2_TYPE_II")
        self.assertIn("total_audit_events", report.summary)


class TestPluginSDK(unittest.TestCase):
    """Test plugin SDK functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.plugin_dir = Path(self.temp_dir) / "plugins"
        self.plugin_dir.mkdir()
        self.plugin_manager = PluginManager(str(self.plugin_dir))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_sample_plugin(self, plugin_name: str):
        """Create a sample plugin for testing."""
        plugin_path = self.plugin_dir / plugin_name
        plugin_path.mkdir()

        # Create manifest
        manifest = {
            "name": plugin_name,
            "version": "1.0.0",
            "description": "Test plugin",
            "entry_point": "main.py",
            "permissions": ["agentnet.test"],
        }

        with open(plugin_path / "plugin.json", "w") as f:
            json.dump(manifest, f)

        # Create implementation
        code = f"""
from agentnet.plugins import Plugin, PluginInfo

class TestPlugin(Plugin):
    @property
    def info(self):
        return PluginInfo(
            name="{plugin_name}",
            version="1.0.0",
            description="Test plugin"
        )
    
    def initialize(self):
        return super().initialize()
"""

        with open(plugin_path / "main.py", "w") as f:
            f.write(code)

        return plugin_path

    def test_plugin_discovery(self):
        """Test plugin discovery functionality."""
        # Create sample plugins
        self.create_sample_plugin("test_plugin_1")
        self.create_sample_plugin("test_plugin_2")

        discovered = self.plugin_manager.discover_plugins()
        self.assertEqual(len(discovered), 2)

        plugin_names = [info.name for info in discovered]
        self.assertIn("test_plugin_1", plugin_names)
        self.assertIn("test_plugin_2", plugin_names)

    def test_plugin_loading(self):
        """Test plugin loading functionality."""
        self.create_sample_plugin("test_plugin")

        success = self.plugin_manager.load_plugin("test_plugin")
        self.assertTrue(success)

        plugin = self.plugin_manager.registry.get_plugin("test_plugin")
        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.status, PluginStatus.LOADED)

    def test_plugin_lifecycle(self):
        """Test plugin lifecycle management."""
        self.create_sample_plugin("lifecycle_test")

        # Load plugin
        self.assertTrue(self.plugin_manager.load_plugin("lifecycle_test"))

        # Initialize plugin
        self.assertTrue(self.plugin_manager.initialize_plugin("lifecycle_test"))
        plugin = self.plugin_manager.registry.get_plugin("lifecycle_test")
        self.assertEqual(plugin.status, PluginStatus.INITIALIZED)

        # Activate plugin
        self.assertTrue(self.plugin_manager.activate_plugin("lifecycle_test"))
        self.assertEqual(plugin.status, PluginStatus.ACTIVE)

        # Deactivate plugin
        self.assertTrue(self.plugin_manager.deactivate_plugin("lifecycle_test"))
        self.assertEqual(plugin.status, PluginStatus.INACTIVE)

    def test_security_policy(self):
        """Test plugin security policy enforcement."""
        security_policy = SecurityPolicy(
            name="test_policy",
            allowed_permissions={"agentnet.test"},
            blocked_permissions={"system.execute"},
        )

        self.plugin_manager.set_security_policy(security_policy)

        # Test with allowed plugin
        plugin_info = PluginInfo(
            name="allowed_plugin", version="1.0.0", permissions=["agentnet.test"]
        )
        self.assertTrue(security_policy.can_load_plugin(plugin_info))

        # Test with blocked plugin
        blocked_plugin_info = PluginInfo(
            name="blocked_plugin", version="1.0.0", permissions=["system.execute"]
        )
        self.assertFalse(security_policy.can_load_plugin(blocked_plugin_info))

    def test_plugin_hooks(self):
        """Test plugin hook system."""
        self.create_sample_plugin("hook_test")

        # Load and activate plugin
        self.plugin_manager.load_plugin("hook_test")
        self.plugin_manager.initialize_plugin("hook_test")
        self.plugin_manager.activate_plugin("hook_test")

        # Test global hook registration
        hook_called = False

        def test_hook(*args, **kwargs):
            nonlocal hook_called
            hook_called = True
            return "hook_result"

        self.plugin_manager.register_global_hook("test_hook", test_hook)
        results = self.plugin_manager.execute_global_hook("test_hook")

        self.assertTrue(hook_called)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "hook_result")


class TestIntegratedP6Features(unittest.TestCase):
    """Test integrated P6 features working together."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Setup components
        audit_storage = AuditStorage(os.path.join(self.temp_dir, "audit.db"))
        self.audit_logger = AuditLogger(audit_storage)
        self.export_manager = ExportControlManager()
        self.plugin_manager = PluginManager(os.path.join(self.temp_dir, "plugins"))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_audit_export_integration(self):
        """Test integration between audit logging and export controls."""
        # Perform export evaluation
        evaluation = self.export_manager.evaluate_export_eligibility(
            "This contains sensitive SSN: 123-45-6789", "external_system"
        )

        # Log the export evaluation
        self.audit_logger.log_export_evaluation(
            classification_level=evaluation["evaluation"]["classification_level"],
            export_allowed=evaluation["evaluation"]["export_allowed"],
            destination="external_system",
        )

        # Verify audit event was created
        audit_storage = self.audit_logger.storage
        events = audit_storage.get_events(
            AuditQuery(event_types=[AuditEventType.EXPORT_EVALUATION])
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, AuditEventType.EXPORT_EVALUATION)

    def test_plugin_audit_integration(self):
        """Test integration between plugins and audit logging."""
        # This would test plugin activities being audited
        # For now, verify the audit logger can handle plugin events

        self.audit_logger.log_event(
            AuditEvent(
                event_type=AuditEventType.CONFIG_CHANGE,
                action="plugin_loaded",
                details={"plugin_name": "test_plugin", "version": "1.0.0"},
            )
        )

        audit_storage = self.audit_logger.storage
        events = audit_storage.get_events(limit=1)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].action, "plugin_loaded")


def run_p6_tests():
    """Run all P6 enterprise hardening tests."""
    print("🧪 Running P6 Enterprise Hardening Tests")
    print("=" * 50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestExportControls,
        TestAuditWorkflow,
        TestPluginSDK,
        TestIntegratedP6Features,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print results
    print("=" * 50)
    if result.wasSuccessful():
        print("✅ All P6 tests passed successfully!")
        print(f"📊 Tests run: {result.testsRun}")
        print("🏢 P6 Enterprise Hardening features are ready for production!")
    else:
        print("❌ Some P6 tests failed")
        print(f"📊 Tests run: {result.testsRun}")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"❌ Errors: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_p6_tests()
    exit(0 if success else 1)
