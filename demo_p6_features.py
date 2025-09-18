#!/usr/bin/env python3
"""
AgentNet P6 Enterprise Hardening Features Demo

Demonstrates the comprehensive P6 features:
- Export Controls: Data classification and content redaction
- Audit Workflow: Comprehensive audit logging and compliance reporting  
- Plugin SDK: Extensible plugin framework with security controls

This demo showcases enterprise-grade security and extensibility features
required for production deployment in regulated environments.
"""

import os
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Import P6 modules
from agentnet.compliance import (
    ExportControlManager, DataClassifier, ContentRedactor, 
    ComplianceReporter, CompliancePolicy, ClassificationLevel
)
from agentnet.audit import (
    AuditWorkflow, AuditLogger, AuditStorage, AuditQuery,
    AuditEvent, AuditEventType, AuditSeverity, AuditDashboard
)
from agentnet.plugins import (
    PluginManager, Plugin, PluginInfo, PluginStatus,
    PluginSandbox, SecurityPolicy, PluginLoader
)

# Import core AgentNet for integration
from agentnet import AgentNet, ExampleEngine, MonitorFactory, MonitorSpec, Severity


def demo_export_controls():
    """Demonstrate P6 Export Control capabilities."""
    print("\n🛡️  P6 Export Control Demo")
    print("-" * 40)
    
    # Create export control manager
    export_manager = ExportControlManager()
    
    # Test content samples
    test_content = [
        "This is public information about our product features.",
        "Contact support at support@company.com for assistance.",
        "User SSN: 123-45-6789 should be protected.",
        "Credit card 4532-1234-5678-9012 requires redaction.",
        "This document contains CONFIDENTIAL information about our API keys.",
        "Export controlled technology under ITAR regulations.",
        "API_KEY=sk_live_123456789abcdef should be secured."
    ]
    
    print("Testing content classification and export control...")
    
    for i, content in enumerate(test_content, 1):
        print(f"\n📄 Content {i}: {content[:50]}...")
        
        # Evaluate export eligibility
        evaluation = export_manager.evaluate_export_eligibility(
            content=content,
            destination="external_partner"
        )
        
        classification_level = evaluation["evaluation"]["classification_level"]
        export_decision = evaluation["export_decision"]
        detections = evaluation["evaluation"]["detections"]
        
        print(f"   📊 Classification: {classification_level}")
        print(f"   🚦 Export Decision: {export_decision}")
        
        if detections:
            print(f"   🔍 Detections: {len(detections)} patterns found")
            for detection in detections[:2]:  # Show first 2
                print(f"      - {detection['type']}: {detection['level']}")
        
        if evaluation["redacted_content"]:
            original_length = evaluation["evaluation"]["original_content_length"]
            redacted_length = evaluation["evaluation"]["redacted_content_length"]
            redaction_ratio = (original_length - redacted_length) / original_length * 100
            if redaction_ratio > 0:
                print(f"   ✂️  Redaction: {redaction_ratio:.1f}% of content redacted")
    
    # Generate compliance report
    print("\n📋 Generating export control compliance report...")
    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = os.path.join(temp_dir, "export_control_report.json")
        export_manager.export_compliance_report(report_path)
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        print(f"   ✅ Total evaluations: {report['summary']['total_evaluations']}")
        print(f"   ✅ Approved exports: {report['summary']['approved_exports']}")
        print(f"   ❌ Denied exports: {report['summary']['denied_exports']}")
        print(f"   ✂️  Redactions performed: {report['summary']['redactions_performed']}")


def demo_audit_workflow():
    """Demonstrate P6 Audit Workflow capabilities."""
    print("\n📋 P6 Audit Workflow Demo")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create audit storage and logger
        storage_path = os.path.join(temp_dir, "audit.db")
        audit_storage = AuditStorage(storage_path)
        audit_logger = AuditLogger(audit_storage)
        
        # Create audit workflow
        audit_workflow = AuditWorkflow(audit_logger)
        
        print("Generating sample audit events...")
        
        # Simulate various audit events
        sample_events = [
            {
                "type": AuditEventType.USER_LOGIN,
                "user_id": "admin_user",
                "details": {"login_method": "sso", "ip_address": "192.168.1.100"}
            },
            {
                "type": AuditEventType.AGENT_INFERENCE,
                "agent_id": "security_agent",
                "session_id": "sess_123",
                "details": {"task": "security_analysis", "duration_ms": 1250}
            },
            {
                "type": AuditEventType.POLICY_VIOLATION,
                "agent_id": "test_agent",
                "session_id": "sess_124",
                "severity": AuditSeverity.HIGH,
                "details": {"violation_type": "content_filter", "rule": "no_pii"}
            },
            {
                "type": AuditEventType.EXPORT_EVALUATION,
                "severity": AuditSeverity.MEDIUM,
                "details": {"classification": "confidential", "destination": "external"}
            },
            {
                "type": AuditEventType.SECURITY_ALERT,
                "severity": AuditSeverity.CRITICAL,
                "details": {"alert_type": "suspicious_activity", "threat_level": "high"}
            }
        ]
        
        # Log events
        for event_data in sample_events:
            event = AuditEvent(
                event_type=event_data["type"],
                severity=event_data.get("severity", AuditSeverity.LOW),
                user_id=event_data.get("user_id"),
                session_id=event_data.get("session_id"),
                agent_id=event_data.get("agent_id"),
                action=f"{event_data['type'].value}_action",
                details=event_data.get("details", {})
            )
            audit_logger.log_event(event)
        
        print(f"   ✅ Logged {len(sample_events)} audit events")
        
        # Query audit events
        print("\n🔍 Querying audit events...")
        
        # Get high-severity events
        high_severity_query = AuditQuery(
            severity_levels=[AuditSeverity.HIGH, AuditSeverity.CRITICAL],
            limit=10
        )
        high_severity_events = audit_storage.get_events(high_severity_query)
        print(f"   🚨 High severity events: {len(high_severity_events)}")
        
        # Get security events
        security_query = AuditQuery(
            event_types=[AuditEventType.SECURITY_ALERT, AuditEventType.POLICY_VIOLATION],
            limit=10
        )
        security_events = audit_storage.get_events(security_query)
        print(f"   🔒 Security events: {len(security_events)}")
        
        # Generate statistics
        print("\n📊 Audit statistics...")
        stats = audit_storage.get_event_statistics()
        print(f"   📈 Total events: {stats['total_events']}")
        print(f"   📊 Events by type: {len(stats['events_by_type'])} types")
        print(f"   ⚠️  Events by severity: {len(stats['events_by_severity'])} levels")
        
        # Generate compliance dashboard
        print("\n📱 Generating compliance dashboard...")
        dashboard = AuditDashboard(audit_storage)
        dashboard_data = dashboard.generate_compliance_dashboard(days=7)
        
        print(f"   🎯 Compliance score: {dashboard_data['summary']['compliance_score']:.1f}%")
        print(f"   🚨 High-risk events: {dashboard_data['summary']['high_risk_events']}")
        print(f"   🔒 Security incidents: {dashboard_data['summary']['security_incidents']}")
        
        # Generate SOC2 dashboard
        soc2_dashboard = dashboard.generate_soc2_dashboard()
        print(f"   🏢 SOC2 compliance score: {soc2_dashboard['overall_compliance_score']:.1f}%")
        
        # Test compliance reporting
        print("\n📄 Generating compliance reports...")
        compliance_reporter = ComplianceReporter(temp_dir)
        
        # Generate SOC2 report
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        soc2_report = compliance_reporter.generate_soc2_report(start_date, end_date)
        
        report_path = compliance_reporter.save_report(soc2_report)
        print(f"   📋 SOC2 report saved: {Path(report_path).name}")
        print(f"   📊 Report covers {soc2_report.summary['total_audit_events']} events")


def demo_plugin_sdk():
    """Demonstrate P6 Plugin SDK capabilities."""
    print("\n🔌 P6 Plugin SDK Demo")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        plugin_dir = Path(temp_dir) / "plugins"
        plugin_dir.mkdir()
        
        # Create a sample plugin
        sample_plugin_dir = plugin_dir / "sample_monitor"
        sample_plugin_dir.mkdir()
        
        # Create plugin manifest
        plugin_manifest = {
            "name": "sample_monitor",
            "version": "1.0.0",
            "description": "Sample monitoring plugin for demo",
            "author": "AgentNet Team",
            "license": "MIT",
            "plugin_type": "monitor",
            "entry_point": "main.py",
            "permissions": ["agentnet.monitor.register"],
            "sandboxed": True,
            "config_schema": {
                "threshold": {"type": "number", "default": 0.8}
            }
        }
        
        with open(sample_plugin_dir / "plugin.json", 'w') as f:
            json.dump(plugin_manifest, f, indent=2)
        
        # Create plugin implementation
        plugin_code = '''
from agentnet.plugins import Plugin, PluginInfo

class SampleMonitorPlugin(Plugin):
    @property
    def info(self):
        return PluginInfo(
            name="sample_monitor",
            version="1.0.0",
            description="Sample monitoring plugin for demo",
            plugin_type="monitor"
        )
    
    def initialize(self):
        print(f"Initializing {self.info.name} plugin")
        return super().initialize()
    
    def activate(self):
        print(f"Activating {self.info.name} plugin")
        self.register_hook("agent_inference", self.monitor_inference)
        return super().activate()
    
    def monitor_inference(self, agent_id, task, result):
        """Monitor agent inference for compliance."""
        print(f"Plugin monitoring inference: {agent_id} - {task[:30]}...")
        return {"plugin_check": "passed", "monitored_by": self.info.name}
'''
        
        with open(sample_plugin_dir / "main.py", 'w') as f:
            f.write(plugin_code)
        
        print("🔧 Created sample plugin structure")
        print(f"   📁 Plugin directory: {sample_plugin_dir.name}")
        
        # Create plugin manager with security policy
        plugin_manager = PluginManager(str(plugin_dir))
        
        # Create security policy
        security_policy = SecurityPolicy(
            name="demo_policy",
            allowed_permissions={"agentnet.monitor.register", "agentnet.agent.create"},
            blocked_imports={"os", "sys", "subprocess"},
            max_memory_mb=128,
            max_cpu_time_seconds=10
        )
        plugin_manager.set_security_policy(security_policy)
        
        print("🔐 Security policy configured")
        print(f"   ✅ Allowed permissions: {len(security_policy.allowed_permissions)}")
        print(f"   ❌ Blocked imports: {len(security_policy.blocked_imports)}")
        
        # Discover plugins
        discovered = plugin_manager.discover_plugins()
        print(f"\n🔍 Plugin discovery completed: {len(discovered)} plugins found")
        
        for plugin_info in discovered:
            print(f"   📦 {plugin_info.name} v{plugin_info.version}")
            print(f"      Type: {plugin_info.plugin_type}")
            print(f"      Sandboxed: {plugin_info.sandboxed}")
        
        # Load and initialize plugin
        if discovered:
            plugin_name = discovered[0].name
            print(f"\n⚡ Loading plugin: {plugin_name}")
            
            success = plugin_manager.load_plugin(plugin_name)
            if success:
                print(f"   ✅ Plugin loaded successfully")
                
                # Initialize plugin
                success = plugin_manager.initialize_plugin(plugin_name)
                if success:
                    print(f"   ✅ Plugin initialized successfully")
                    
                    # Activate plugin
                    success = plugin_manager.activate_plugin(plugin_name)
                    if success:
                        print(f"   ✅ Plugin activated successfully")
                        
                        # Test plugin hook
                        print(f"\n🎣 Testing plugin hooks...")
                        results = plugin_manager.execute_global_hook(
                            "agent_inference",
                            agent_id="test_agent",
                            task="Test inference task for plugin demo",
                            result={"content": "Test result"}
                        )
                        print(f"   🎯 Hook execution results: {len(results)} callbacks executed")
                        
                        if results:
                            for result in results:
                                print(f"      📊 Result: {result}")
        
        # Display system status
        print(f"\n📊 Plugin system status:")
        system_status = plugin_manager.get_system_status()
        print(f"   📦 Total plugins: {system_status['total_plugins']}")
        print(f"   📈 Status counts: {system_status['status_counts']}")
        print(f"   🎣 Global hooks: {len(system_status['global_hooks'])}")


def demo_integrated_p6_features():
    """Demonstrate integrated P6 features working together."""
    print("\n🏢 P6 Integrated Enterprise Features Demo")
    print("-" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup integrated environment
        audit_storage = AuditStorage(os.path.join(temp_dir, "audit.db"))
        audit_logger = AuditLogger(audit_storage)
        export_manager = ExportControlManager()
        plugin_manager = PluginManager(os.path.join(temp_dir, "plugins"))
        
        print("🏗️  Enterprise environment initialized")
        print("   📋 Audit logging: Active")
        print("   🛡️  Export controls: Active") 
        print("   🔌 Plugin system: Active")
        
        # Simulate enterprise workflow
        print(f"\n🏢 Simulating enterprise agent workflow...")
        
        # Create agent with monitoring
        engine = ExampleEngine()
        
        # Create monitors
        monitors = [
            MonitorFactory.build(MonitorSpec(
                name="enterprise_content_filter",
                type="keyword",
                severity="severe",
                params={"keywords": ["confidential", "secret", "proprietary"]}
            )),
            MonitorFactory.build(MonitorSpec(
                name="pii_detector", 
                type="regex",
                severity="severe",
                params={"pattern": r"\b\d{3}-\d{2}-\d{4}\b"}
            ))
        ]
        
        agent = AgentNet(
            name="enterprise_agent",
            engine=engine,
            system_prompt="You are an enterprise AI assistant with strict compliance requirements.",
            style="professional",
            monitors=monitors
        )
        
        # Log agent creation
        audit_logger.log_event(AuditEvent(
            event_type=AuditEventType.AGENT_CREATED,
            agent_id=agent.name,
            action="enterprise_agent_created",
            details={"agent_type": "enterprise", "monitors": len(agent.monitors)}
        ))
        
        # Test scenarios with integrated features
        test_scenarios = [
            "Analyze the quarterly financial report for strategic insights.",
            "Process customer data including SSN 123-45-6789 for verification.",
            "Generate a CONFIDENTIAL summary of our competitive analysis.",
            "Create export documentation for international technology transfer."
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario[:40]}...")
            
            # Agent inference with audit logging
            audit_logger.log_event(AuditEvent(
                event_type=AuditEventType.AGENT_INFERENCE,
                agent_id=agent.name,
                session_id=f"enterprise_session_{i}",
                action="agent_inference_started",
                details={"scenario": scenario}
            ))
            
            # Generate response
            try:
                result = agent.generate_reasoning_tree(scenario)
                content = result.get("result", {}).get("content", "")
                
                # Check for violations
                violations = result.get("violations", [])
                if violations:
                    audit_logger.log_policy_violation(
                        violation_details=violations[0],
                        session_id=f"enterprise_session_{i}",
                        agent_id=agent.name
                    )
                    print(f"   ⚠️  Policy violations detected: {len(violations)}")
                
                # Export control evaluation
                export_eval = export_manager.evaluate_export_eligibility(
                    content=content,
                    destination="partner_system"
                )
                
                audit_logger.log_export_evaluation(
                    classification_level=export_eval["evaluation"]["classification_level"],
                    export_allowed=export_eval["evaluation"]["export_allowed"],
                    destination="partner_system"
                )
                
                print(f"   📊 Classification: {export_eval['evaluation']['classification_level']}")
                print(f"   🚦 Export: {export_eval['export_decision']}")
                
            except Exception as e:
                audit_logger.log_event(AuditEvent(
                    event_type=AuditEventType.SYSTEM_ERROR,
                    severity=AuditSeverity.HIGH,
                    agent_id=agent.name,
                    action="agent_inference_failed",
                    details={"error": str(e), "scenario": scenario}
                ))
                print(f"   ❌ Error: {str(e)[:50]}...")
        
        # Generate comprehensive compliance report
        print(f"\n📊 Generating comprehensive compliance report...")
        
        stats = audit_storage.get_event_statistics()
        export_trail = export_manager.get_export_audit_trail()
        
        compliance_summary = {
            "audit_events": stats["total_events"],
            "export_evaluations": len(export_trail),
            "policy_violations": stats["events_by_type"].get("policy_violation", 0),
            "security_incidents": stats["events_by_type"].get("security_alert", 0),
            "compliance_score": max(0, 100 - stats["events_by_type"].get("policy_violation", 0) * 10)
        }
        
        print(f"   📈 Audit events: {compliance_summary['audit_events']}")
        print(f"   🛡️  Export evaluations: {compliance_summary['export_evaluations']}")
        print(f"   ⚠️  Policy violations: {compliance_summary['policy_violations']}")
        print(f"   🎯 Overall compliance score: {compliance_summary['compliance_score']:.1f}%")
        
        # Enterprise recommendations
        print(f"\n💡 Enterprise recommendations:")
        if compliance_summary['policy_violations'] > 0:
            print("   • Review and update content filtering policies")
        if compliance_summary['export_evaluations'] > 5:
            print("   • Implement additional export control training")
        print("   • Maintain regular compliance monitoring")
        print("   • Schedule periodic security audits")


def main():
    """Main demo function."""
    print("=" * 60)
    print("🏢 AgentNet P6 Enterprise Hardening Features Demo")
    print("=" * 60)
    print("Demonstrating comprehensive enterprise security and extensibility:")
    print("• Export Controls: Data classification and content redaction")
    print("• Audit Workflow: Compliance logging and SOC2 reporting")
    print("• Plugin SDK: Secure extensible plugin framework")
    print("=" * 60)
    
    try:
        # Individual feature demos
        demo_export_controls()
        demo_audit_workflow()
        demo_plugin_sdk()
        
        # Integrated demo
        demo_integrated_p6_features()
        
        print("\n" + "=" * 60)
        print("✅ P6 Enterprise Hardening Demo Completed Successfully!")
        print("🏢 AgentNet is now ready for enterprise deployment with:")
        print("   🛡️  Comprehensive export control capabilities")
        print("   📋 SOC2-compliant audit workflow infrastructure") 
        print("   🔌 Secure and extensible plugin framework")
        print("   📊 Enterprise-grade compliance reporting")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()