#!/usr/bin/env python3
"""
Simple P6 Enterprise Hardening Demo

A focused demonstration of the core P6 features:
- Export Controls: Data classification and redaction
- Audit Workflow: Enterprise logging and compliance
- Plugin SDK: Secure extensible framework

This demo shows the essential enterprise capabilities working together.
"""

import tempfile
import os
from datetime import datetime, timedelta

# Import P6 modules
from agentnet.compliance import ExportControlManager, ComplianceReporter, ClassificationLevel
from agentnet.audit import AuditLogger, AuditStorage, AuditEvent, AuditEventType, AuditSeverity, AuditDashboard
from agentnet.plugins import PluginManager, SecurityPolicy


def demo_export_controls():
    """Demonstrate export control capabilities."""
    print("🛡️  Export Controls Demo")
    print("-" * 30)
    
    export_manager = ExportControlManager()
    
    # Test different types of sensitive content
    test_cases = [
        ("Public info", "Our product offers great features for customers."),
        ("Email address", "Contact us at support@company.com for help."),
        ("SSN data", "Customer SSN: 123-45-6789 requires protection."),
        ("Confidential", "This CONFIDENTIAL document contains trade secrets."),
        ("API credentials", "Use API_KEY=sk_live_abcd1234 for authentication.")
    ]
    
    for name, content in test_cases[:3]:  # Limit for demo
        print(f"\n📄 {name}: {content[:40]}...")
        
        evaluation = export_manager.evaluate_export_eligibility(content, "external_partner")
        
        print(f"   📊 Classification: {evaluation['evaluation']['classification_level']}")
        print(f"   🚦 Export: {evaluation['export_decision']}")
        
        if evaluation['evaluation']['detections']:
            det_count = len(evaluation['evaluation']['detections'])
            print(f"   🔍 {det_count} sensitive pattern(s) detected")


def demo_audit_workflow():
    """Demonstrate audit workflow capabilities."""
    print("\n📋 Audit Workflow Demo")
    print("-" * 30)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup audit system
        storage = AuditStorage(os.path.join(temp_dir, "audit.db"))
        audit_logger = AuditLogger(storage)
        
        # Log some enterprise events
        events = [
            AuditEvent(
                event_type=AuditEventType.USER_LOGIN,
                user_id="admin_user",
                action="sso_login",
                details={"method": "saml", "ip": "192.168.1.100"}
            ),
            AuditEvent(
                event_type=AuditEventType.POLICY_VIOLATION,
                severity=AuditSeverity.HIGH,
                agent_id="content_agent",
                action="content_filter_violation",
                details={"rule": "pii_detection", "content_type": "ssn"}
            ),
            AuditEvent(
                event_type=AuditEventType.EXPORT_EVALUATION,
                severity=AuditSeverity.MEDIUM,
                action="export_denied",
                details={"classification": "confidential", "destination": "external"}
            )
        ]
        
        for event in events:
            audit_logger.log_event(event)
        
        print(f"✅ Logged {len(events)} audit events")
        
        # Generate compliance metrics
        stats = storage.get_event_statistics()
        print(f"📊 Total events: {stats['total_events']}")
        print(f"⚠️  Event types: {len(stats['events_by_type'])}")
        
        # Create compliance dashboard
        dashboard = AuditDashboard(storage)
        dashboard_data = dashboard.generate_compliance_dashboard(days=1)
        
        print(f"🎯 Compliance score: {dashboard_data['summary']['compliance_score']:.1f}%")
        print(f"🚨 High-risk events: {dashboard_data['summary']['high_risk_events']}")


def demo_plugin_sdk():
    """Demonstrate plugin SDK capabilities.""" 
    print("\n🔌 Plugin SDK Demo")
    print("-" * 30)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create plugin manager with security policy
        plugin_manager = PluginManager(os.path.join(temp_dir, "plugins"))
        
        # Configure enterprise security policy
        security_policy = SecurityPolicy(
            name="enterprise_policy",
            allowed_permissions={"agentnet.monitor.register", "agentnet.tool.register"},
            blocked_permissions={"system.execute", "network.raw_socket"},
            blocked_imports={"os", "sys", "subprocess", "socket"},
            max_memory_mb=256,
            max_cpu_time_seconds=30,
            network_access=False
        )
        
        plugin_manager.set_security_policy(security_policy)
        
        print("🔐 Security policy configured:")
        print(f"   ✅ Allowed permissions: {len(security_policy.allowed_permissions)}")
        print(f"   ❌ Blocked permissions: {len(security_policy.blocked_permissions)}")
        print(f"   🚫 Blocked imports: {len(security_policy.blocked_imports)}")
        print(f"   💾 Memory limit: {security_policy.max_memory_mb}MB")
        
        # Test global hook system
        hook_results = []
        
        def test_hook(*args, **kwargs):
            hook_results.append("enterprise_hook_executed")
            return {"status": "success", "plugin": "enterprise_monitor"}
        
        plugin_manager.register_global_hook("agent_inference", test_hook)
        
        # Execute hook
        results = plugin_manager.execute_global_hook(
            "agent_inference", 
            agent_id="test_agent",
            task="Enterprise compliance check"
        )
        
        print(f"🎣 Hook system test: {len(results)} hooks executed")
        if results:
            print(f"   📊 Result: {results[0]['status']}")


def demo_integrated_enterprise():
    """Demonstrate integrated P6 features."""
    print("\n🏢 Integrated Enterprise Demo")
    print("-" * 30)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup integrated enterprise environment
        export_manager = ExportControlManager()
        storage = AuditStorage(os.path.join(temp_dir, "enterprise_audit.db"))
        audit_logger = AuditLogger(storage)
        plugin_manager = PluginManager(os.path.join(temp_dir, "plugins"))
        
        print("🏗️  Enterprise environment initialized")
        
        # Simulate enterprise workflow
        sensitive_content = "Employee record: John Doe, SSN: 123-45-6789, Salary: $75,000"
        
        # 1. Export control evaluation
        export_eval = export_manager.evaluate_export_eligibility(
            sensitive_content, 
            "hr_system"
        )
        
        # 2. Log the export evaluation
        audit_logger.log_export_evaluation(
            classification_level=export_eval['evaluation']['classification_level'],
            export_allowed=export_eval['evaluation']['export_allowed'],
            destination="hr_system",
            content_length=len(sensitive_content)
        )
        
        # 3. Log additional enterprise event
        audit_logger.log_event(AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            user_id="hr_manager",
            action="employee_record_access",
            details={
                "record_type": "employee",
                "classification": export_eval['evaluation']['classification_level'],
                "access_reason": "payroll_processing"
            }
        ))
        
        # 4. Generate enterprise compliance report
        compliance_reporter = ComplianceReporter(temp_dir)
        
        # Record audit events for compliance reporting
        compliance_reporter.record_audit_event(
            "data_export",
            {
                "classification": export_eval['evaluation']['classification_level'],
                "destination": "hr_system",
                "export_allowed": export_eval['evaluation']['export_allowed']
            }
        )
        
        # Generate SOC2 report
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow()
        soc2_report = compliance_reporter.generate_soc2_report(start_date, end_date)
        
        print(f"📊 Enterprise workflow completed:")
        print(f"   🛡️  Content classified as: {export_eval['evaluation']['classification_level']}")
        print(f"   🚦 Export decision: {export_eval['export_decision']}")
        print(f"   📋 Audit events logged: 2")
        print(f"   🏢 SOC2 compliance score: {soc2_report.summary.get('privacy_compliance_rate', 100):.1f}%")
        
        return {
            'export_evaluations': 1,
            'audit_events': 2,
            'compliance_score': soc2_report.summary.get('privacy_compliance_rate', 100),
            'classification_level': export_eval['evaluation']['classification_level']
        }


def main():
    """Main demo function."""
    print("=" * 50)
    print("🏢 AgentNet P6 Enterprise Hardening Demo")
    print("=" * 50)
    print("Enterprise-grade security and compliance features:")
    print("• Export Controls: Data classification & redaction")
    print("• Audit Workflow: SOC2 compliance & reporting")
    print("• Plugin SDK: Secure extensible framework")
    print("=" * 50)
    
    try:
        # Run individual demos
        demo_export_controls()
        demo_audit_workflow()
        demo_plugin_sdk()
        
        # Run integrated demo
        enterprise_results = demo_integrated_enterprise()
        
        # Summary
        print("\n" + "=" * 50)
        print("✅ P6 Enterprise Hardening Demo Completed!")
        print("=" * 50)
        print("🏢 Enterprise Capabilities Demonstrated:")
        print(f"   🛡️  Export control evaluations: {enterprise_results['export_evaluations']}")
        print(f"   📋 Audit events processed: {enterprise_results['audit_events']}")
        print(f"   🎯 Compliance score: {enterprise_results['compliance_score']:.1f}%")
        print(f"   📊 Content classification: {enterprise_results['classification_level']}")
        print()
        print("🎉 AgentNet is now enterprise-ready with:")
        print("   • Comprehensive export control system")
        print("   • SOC2-compliant audit infrastructure")
        print("   • Secure and extensible plugin framework")
        print("   • Production-grade compliance reporting")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()