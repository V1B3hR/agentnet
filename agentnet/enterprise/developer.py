"""
Developer Platform Components for AgentNet Phase 8

Provides tools for the developer ecosystem:
- Visual agent workflow designer (web-based GUI)
- Low-code/no-code agent creation interface
- Agent marketplace with verified community plugins
- IDE extensions for agent development
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WorkflowNode:
    """Represents a node in the visual workflow designer."""
    id: str
    type: str  # agent, tool, condition, merger, etc.
    name: str
    config: Dict[str, Any]
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})
    connections: List[str] = field(default_factory=list)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition for the visual designer."""
    id: str
    name: str
    description: str
    nodes: List[WorkflowNode]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"


@dataclass
class AgentTemplate:
    """Template definition for low-code agent creation."""
    id: str
    name: str
    description: str
    category: str
    parameters: List[Dict[str, Any]]
    template_code: str
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class MarketplacePlugin:
    """Plugin definition for the agent marketplace."""
    id: str
    name: str
    description: str
    author: str
    version: str
    category: str
    verified: bool = False
    downloads: int = 0
    rating: float = 0.0
    source_url: Optional[str] = None
    documentation_url: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class WorkflowDesigner:
    """Visual agent workflow designer (web-based GUI foundation)."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./workflows")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workflows: Dict[str, WorkflowDefinition] = {}
    
    def create_workflow(self, name: str, description: str = "") -> WorkflowDefinition:
        """Create a new workflow definition."""
        workflow_id = f"workflow_{datetime.now().isoformat()}_{hash(name) % 10000}"
        workflow = WorkflowDefinition(
            id=workflow_id,
            name=name,
            description=description,
            nodes=[]
        )
        self.workflows[workflow_id] = workflow
        self.logger.info(f"Created workflow: {name} (ID: {workflow_id})")
        return workflow
    
    def add_node(self, workflow_id: str, node_type: str, name: str, 
                 config: Dict[str, Any], position: Dict[str, float] = None) -> str:
        """Add a node to a workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        node_id = f"node_{len(self.workflows[workflow_id].nodes)}_{hash(name) % 1000}"
        node = WorkflowNode(
            id=node_id,
            type=node_type,
            name=name,
            config=config,
            position=position or {"x": 0, "y": 0}
        )
        
        self.workflows[workflow_id].nodes.append(node)
        self.logger.info(f"Added node {name} to workflow {workflow_id}")
        return node_id
    
    def connect_nodes(self, workflow_id: str, from_node_id: str, to_node_id: str):
        """Connect two nodes in a workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        from_node = next((n for n in workflow.nodes if n.id == from_node_id), None)
        
        if not from_node:
            raise ValueError(f"Node {from_node_id} not found in workflow")
        
        if to_node_id not in from_node.connections:
            from_node.connections.append(to_node_id)
            self.logger.info(f"Connected {from_node_id} -> {to_node_id} in workflow {workflow_id}")
    
    def validate_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Validate a workflow definition."""
        if workflow_id not in self.workflows:
            return {"valid": False, "errors": ["Workflow not found"]}
        
        workflow = self.workflows[workflow_id]
        errors = []
        
        # Check for cycles
        if self._has_cycles(workflow):
            errors.append("Workflow contains cycles")
        
        # Check for orphaned nodes
        orphaned = self._find_orphaned_nodes(workflow)
        if orphaned:
            errors.append(f"Orphaned nodes found: {orphaned}")
        
        # Check for missing node types
        invalid_types = self._check_node_types(workflow)
        if invalid_types:
            errors.append(f"Invalid node types: {invalid_types}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def export_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Export workflow definition to JSON-serializable format."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "version": workflow.version,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type,
                    "name": node.name,
                    "config": node.config,
                    "position": node.position,
                    "connections": node.connections
                }
                for node in workflow.nodes
            ],
            "metadata": workflow.metadata,
            "created_at": workflow.created_at.isoformat()
        }
    
    def import_workflow(self, workflow_data: Dict[str, Any]) -> str:
        """Import workflow definition from JSON data."""
        workflow = WorkflowDefinition(
            id=workflow_data["id"],
            name=workflow_data["name"],
            description=workflow_data["description"],
            version=workflow_data.get("version", "1.0.0"),
            nodes=[
                WorkflowNode(
                    id=node_data["id"],
                    type=node_data["type"],
                    name=node_data["name"],
                    config=node_data["config"],
                    position=node_data.get("position", {"x": 0, "y": 0}),
                    connections=node_data.get("connections", [])
                )
                for node_data in workflow_data["nodes"]
            ],
            metadata=workflow_data.get("metadata", {}),
            created_at=datetime.fromisoformat(workflow_data.get("created_at", datetime.now().isoformat()))
        )
        
        self.workflows[workflow.id] = workflow
        self.logger.info(f"Imported workflow: {workflow.name}")
        return workflow.id
    
    def save_workflow(self, workflow_id: str):
        """Save workflow to disk."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow_data = self.export_workflow(workflow_id)
        file_path = self.storage_path / f"{workflow_id}.json"
        
        with open(file_path, "w") as f:
            json.dump(workflow_data, f, indent=2)
        
        self.logger.info(f"Saved workflow {workflow_id} to {file_path}")
    
    def load_workflow(self, workflow_id: str) -> str:
        """Load workflow from disk."""
        file_path = self.storage_path / f"{workflow_id}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {file_path}")
        
        with open(file_path, "r") as f:
            workflow_data = json.load(f)
        
        return self.import_workflow(workflow_data)
    
    def _has_cycles(self, workflow: WorkflowDefinition) -> bool:
        """Check if workflow has cycles using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = next((n for n in workflow.nodes if n.id == node_id), None)
            if node:
                for connection in node.connections:
                    if connection not in visited:
                        if dfs(connection):
                            return True
                    elif connection in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node in workflow.nodes:
            if node.id not in visited:
                if dfs(node.id):
                    return True
        
        return False
    
    def _find_orphaned_nodes(self, workflow: WorkflowDefinition) -> List[str]:
        """Find nodes with no incoming connections (except start nodes)."""
        all_targets = set()
        for node in workflow.nodes:
            all_targets.update(node.connections)
        
        orphaned = []
        for node in workflow.nodes:
            if node.id not in all_targets and node.type != "start":
                orphaned.append(node.id)
        
        return orphaned
    
    def _check_node_types(self, workflow: WorkflowDefinition) -> List[str]:
        """Check for invalid node types."""
        valid_types = {"start", "end", "agent", "tool", "condition", "merger", "splitter"}
        invalid = []
        
        for node in workflow.nodes:
            if node.type not in valid_types:
                invalid.append(node.type)
        
        return list(set(invalid))


class LowCodeInterface:
    """Low-code/no-code agent creation interface."""
    
    def __init__(self, templates_path: Optional[str] = None):
        self.templates_path = Path(templates_path) if templates_path else Path("./templates")
        self.templates_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.templates: Dict[str, AgentTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default agent templates."""
        # Customer Service Agent Template
        self.add_template(AgentTemplate(
            id="customer_service",
            name="Customer Service Agent",
            description="Handles customer inquiries and support requests",
            category="support",
            parameters=[
                {"name": "company_name", "type": "string", "required": True, "description": "Name of your company"},
                {"name": "knowledge_base", "type": "string", "required": False, "description": "Path to knowledge base"},
                {"name": "escalation_threshold", "type": "number", "default": 3, "description": "Number of attempts before escalation"}
            ],
            template_code="""
class CustomerServiceAgent(AgentNet):
    def __init__(self, company_name, knowledge_base=None, escalation_threshold=3):
        super().__init__(
            name=f"{company_name} Support Agent",
            personality={"helpfulness": 0.9, "patience": 0.8, "professionalism": 0.9},
            policies=["customer_service_policy"]
        )
        self.company_name = company_name
        self.knowledge_base = knowledge_base
        self.escalation_threshold = escalation_threshold
    
    async def handle_inquiry(self, inquiry):
        # Process customer inquiry
        response = await self.generate_response(inquiry)
        return response
""",
            examples=[
                {"name": "Basic Setup", "code": "agent = CustomerServiceAgent('Acme Corp')"},
                {"name": "With Knowledge Base", "code": "agent = CustomerServiceAgent('Acme Corp', 'kb.json')"}
            ],
            tags=["support", "customer", "service"]
        ))
        
        # Data Analysis Agent Template
        self.add_template(AgentTemplate(
            id="data_analyst",
            name="Data Analysis Agent",
            description="Performs data analysis and generates insights",
            category="analytics",
            parameters=[
                {"name": "data_sources", "type": "list", "required": True, "description": "List of data sources"},
                {"name": "analysis_type", "type": "string", "default": "descriptive", "description": "Type of analysis"},
                {"name": "output_format", "type": "string", "default": "report", "description": "Output format"}
            ],
            template_code="""
class DataAnalysisAgent(AgentNet):
    def __init__(self, data_sources, analysis_type="descriptive", output_format="report"):
        super().__init__(
            name="Data Analyst",
            personality={"analytical": 0.9, "detail_oriented": 0.8, "objective": 0.9},
            tools=["data_processor", "chart_generator", "statistics"]
        )
        self.data_sources = data_sources
        self.analysis_type = analysis_type
        self.output_format = output_format
    
    async def analyze_data(self, query):
        # Perform data analysis
        results = await self.process_with_tools(query)
        return results
""",
            examples=[
                {"name": "Sales Analysis", "code": "agent = DataAnalysisAgent(['sales.csv', 'customers.json'])"},
                {"name": "Predictive Analysis", "code": "agent = DataAnalysisAgent(['data.csv'], 'predictive')"}
            ],
            tags=["analytics", "data", "insights"]
        ))
    
    def add_template(self, template: AgentTemplate):
        """Add a new agent template."""
        self.templates[template.id] = template
        self.logger.info(f"Added template: {template.name}")
    
    def get_template(self, template_id: str) -> Optional[AgentTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self, category: Optional[str] = None) -> List[AgentTemplate]:
        """List available templates, optionally filtered by category."""
        templates = list(self.templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates
    
    def generate_agent_code(self, template_id: str, parameters: Dict[str, Any]) -> str:
        """Generate agent code from template and parameters."""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Simple template parameter substitution
        code = template.template_code
        for param in template.parameters:
            param_name = param["name"]
            if param_name in parameters:
                value = parameters[param_name]
                if isinstance(value, str):
                    value = f'"{value}"'
                code = code.replace(f"{{{param_name}}}", str(value))
        
        return code
    
    def validate_parameters(self, template_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against template requirements."""
        template = self.get_template(template_id)
        if not template:
            return {"valid": False, "errors": ["Template not found"]}
        
        errors = []
        for param in template.parameters:
            param_name = param["name"]
            if param.get("required", False) and param_name not in parameters:
                errors.append(f"Required parameter missing: {param_name}")
            
            if param_name in parameters:
                value = parameters[param_name]
                param_type = param["type"]
                
                if param_type == "string" and not isinstance(value, str):
                    errors.append(f"Parameter {param_name} must be a string")
                elif param_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Parameter {param_name} must be a number")
                elif param_type == "list" and not isinstance(value, list):
                    errors.append(f"Parameter {param_name} must be a list")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def create_agent_from_template(self, template_id: str, parameters: Dict[str, Any]) -> str:
        """Create agent code from template with validation."""
        validation = self.validate_parameters(template_id, parameters)
        if not validation["valid"]:
            raise ValueError(f"Parameter validation failed: {validation['errors']}")
        
        return self.generate_agent_code(template_id, parameters)


class AgentMarketplace:
    """Agent marketplace with verified community plugins."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./marketplace")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.plugins: Dict[str, MarketplacePlugin] = {}
        self._load_featured_plugins()
    
    def _load_featured_plugins(self):
        """Load featured/default plugins."""
        # Web Scraper Plugin
        self.add_plugin(MarketplacePlugin(
            id="web_scraper",
            name="Web Scraper Tool",
            description="Extract data from websites with rate limiting and respect for robots.txt",
            author="AgentNet Team",
            version="1.2.0",
            category="tools",
            verified=True,
            rating=4.8,
            downloads=15420,
            source_url="https://github.com/agentnet/plugins/web-scraper",
            documentation_url="https://docs.agentnet.dev/plugins/web-scraper",
            dependencies=["requests", "beautifulsoup4"],
            tags=["web", "scraping", "data-extraction"]
        ))
        
        # Email Integration Plugin
        self.add_plugin(MarketplacePlugin(
            id="email_integration",
            name="Email Integration",
            description="Send and receive emails through various providers (Gmail, Outlook, etc.)",
            author="Community Contributor",
            version="2.1.0",
            category="integrations",
            verified=True,
            rating=4.6,
            downloads=8932,
            source_url="https://github.com/agentnet-community/email-plugin",
            dependencies=["imaplib", "smtplib"],
            tags=["email", "communication", "integration"]
        ))
        
        # Database Connector Plugin
        self.add_plugin(MarketplacePlugin(
            id="database_connector",
            name="Universal Database Connector",
            description="Connect to various databases (PostgreSQL, MySQL, MongoDB, etc.)",
            author="Database Team",
            version="3.0.1",
            category="data",
            verified=True,
            rating=4.9,
            downloads=22156,
            dependencies=["SQLAlchemy", "pymongo", "psycopg2"],
            tags=["database", "sql", "nosql", "data"]
        ))
    
    def add_plugin(self, plugin: MarketplacePlugin):
        """Add a plugin to the marketplace."""
        self.plugins[plugin.id] = plugin
        self.logger.info(f"Added plugin: {plugin.name} v{plugin.version}")
    
    def get_plugin(self, plugin_id: str) -> Optional[MarketplacePlugin]:
        """Get a plugin by ID."""
        return self.plugins.get(plugin_id)
    
    def search_plugins(self, query: str = "", category: Optional[str] = None, 
                      verified_only: bool = False) -> List[MarketplacePlugin]:
        """Search for plugins by query and filters."""
        plugins = list(self.plugins.values())
        
        if query:
            query_lower = query.lower()
            plugins = [
                p for p in plugins 
                if query_lower in p.name.lower() 
                or query_lower in p.description.lower()
                or any(query_lower in tag.lower() for tag in p.tags)
            ]
        
        if category:
            plugins = [p for p in plugins if p.category == category]
        
        if verified_only:
            plugins = [p for p in plugins if p.verified]
        
        # Sort by rating and downloads
        plugins.sort(key=lambda p: (p.rating, p.downloads), reverse=True)
        
        return plugins
    
    def get_popular_plugins(self, limit: int = 10) -> List[MarketplacePlugin]:
        """Get most popular plugins by downloads."""
        plugins = list(self.plugins.values())
        plugins.sort(key=lambda p: p.downloads, reverse=True)
        return plugins[:limit]
    
    def get_top_rated_plugins(self, limit: int = 10) -> List[MarketplacePlugin]:
        """Get top-rated plugins."""
        plugins = list(self.plugins.values())
        plugins.sort(key=lambda p: p.rating, reverse=True)
        return plugins[:limit]
    
    def get_categories(self) -> List[str]:
        """Get all available plugin categories."""
        categories = set(plugin.category for plugin in self.plugins.values())
        return sorted(list(categories))
    
    def install_plugin(self, plugin_id: str) -> Dict[str, Any]:
        """Simulate plugin installation (returns installation info)."""
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            return {"success": False, "error": "Plugin not found"}
        
        # Increment download count
        plugin.downloads += 1
        
        self.logger.info(f"Installing plugin: {plugin.name} v{plugin.version}")
        
        return {
            "success": True,
            "plugin": plugin,
            "dependencies": plugin.dependencies,
            "installation_notes": f"Plugin {plugin.name} installed successfully"
        }
    
    def rate_plugin(self, plugin_id: str, rating: float) -> bool:
        """Rate a plugin (1-5 stars)."""
        if not 1 <= rating <= 5:
            return False
        
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            return False
        
        # Simple rating update (in real implementation, would track individual ratings)
        plugin.rating = (plugin.rating + rating) / 2
        self.logger.info(f"Rated plugin {plugin.name}: {rating} stars")
        return True


class IDEExtension:
    """IDE extension templates and scaffolding for agent development."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.supported_ides = ["vscode", "jetbrains", "vim", "emacs"]
    
    def generate_vscode_extension(self, extension_name: str) -> Dict[str, str]:
        """Generate VSCode extension scaffolding."""
        # Convert extension name to package-friendly format
        package_name = f"agentnet-{extension_name.lower().replace(' ', '-')}"
        
        package_json = {
            "name": package_name,
            "displayName": f"AgentNet {extension_name}",
            "description": f"AgentNet development tools for {extension_name}",
            "version": "0.0.1",
            "engines": {"vscode": "^1.60.0"},
            "categories": ["Other"],
            "activationEvents": ["onLanguage:agentnet"],
            "main": "./out/extension.js",
            "contributes": {
                "languages": [{
                    "id": "agentnet",
                    "aliases": ["AgentNet", "agentnet"],
                    "extensions": [".agent", ".agentnet"]
                }],
                "commands": [{
                    "command": "agentnet.createAgent",
                    "title": "Create New Agent"
                }],
                "snippets": [{
                    "language": "python",
                    "path": "./snippets/agentnet.json"
                }]
            },
            "scripts": {
                "vscode:prepublish": "npm run compile",
                "compile": "tsc -p ./"
            },
            "devDependencies": {
                "@types/vscode": "^1.60.0",
                "typescript": "^4.4.4"
            }
        }
        
        extension_ts = """
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    // Register AgentNet commands
    let createAgent = vscode.commands.registerCommand('agentnet.createAgent', () => {
        // Create new agent workflow
        vscode.window.showInformationMessage('Creating new AgentNet agent...');
    });
    
    context.subscriptions.push(createAgent);
    
    // Register language features
    const provider = vscode.languages.registerCompletionItemProvider('python', {
        provideCompletionItems(document: vscode.TextDocument, position: vscode.Position) {
            // Provide AgentNet code completion
            const completions = [];
            
            // Add AgentNet-specific completions
            const agentCompletion = new vscode.CompletionItem('AgentNet');
            agentCompletion.insertText = new vscode.SnippetString('AgentNet("${1:name}", {"${2:personality}": ${3:0.8}})');
            completions.push(agentCompletion);
            
            return completions;
        }
    });
    
    context.subscriptions.push(provider);
}

export function deactivate() {}
"""
        
        snippets_json = {
            "AgentNet Class": {
                "prefix": "agentnet",
                "body": [
                    "class ${1:AgentName}(AgentNet):",
                    "    def __init__(self, name='${2:Agent}', personality=None):",
                    "        super().__init__(",
                    "            name=name,",
                    "            personality=personality or {'${3:trait}': ${4:0.8}}",
                    "        )",
                    "        $0"
                ],
                "description": "Create new AgentNet agent class"
            },
            "Agent Generation": {
                "prefix": "generate",
                "body": [
                    "result = await agent.generate_reasoning_tree('${1:prompt}')",
                    "$0"
                ],
                "description": "Generate reasoning tree"
            }
        }
        
        return {
            "package.json": json.dumps(package_json, indent=2),
            "src/extension.ts": extension_ts,
            "snippets/agentnet.json": json.dumps(snippets_json, indent=2)
        }
    
    def generate_jetbrains_plugin(self, plugin_name: str) -> Dict[str, str]:
        """Generate JetBrains plugin scaffolding."""
        plugin_xml = f"""
<idea-plugin>
    <id>com.agentnet.{plugin_name.lower()}</id>
    <name>AgentNet {plugin_name}</name>
    <version>1.0</version>
    <vendor email="support@agentnet.dev" url="https://agentnet.dev">AgentNet</vendor>
    
    <description><![CDATA[
        AgentNet development tools for IntelliJ IDEA and other JetBrains IDEs.
        Provides syntax highlighting, code completion, and debugging tools for AgentNet agents.
    ]]></description>
    
    <depends>com.intellij.modules.platform</depends>
    <depends>com.intellij.modules.python</depends>
    
    <extensions defaultExtensionNamed="com.intellij">
        <fileType name="AgentNet" 
                  implementationClass="com.agentnet.AgentNetFileType"
                  fieldName="INSTANCE" 
                  language="AgentNet" 
                  extensions="agent;agentnet"/>
        
        <lang.parserDefinition language="AgentNet"
                              implementationClass="com.agentnet.AgentNetParserDefinition"/>
        
        <completion.contributor language="Python"
                               implementationClass="com.agentnet.AgentNetCompletionContributor"/>
    </extensions>
    
    <actions>
        <action id="agentnet.CreateAgent" 
                class="com.agentnet.CreateAgentAction" 
                text="Create AgentNet Agent">
            <add-to-group group-id="NewGroup" anchor="after" relative-to-action="NewFile"/>
        </action>
    </actions>
</idea-plugin>
"""
        
        java_action = """
package com.agentnet;

import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.ui.Messages;

public class CreateAgentAction extends AnAction {
    @Override
    public void actionPerformed(AnActionEvent e) {
        Messages.showInfoMessage("Creating new AgentNet agent...", "AgentNet");
        // TODO: Open agent creation dialog
    }
}
"""
        
        return {
            "META-INF/plugin.xml": plugin_xml,
            "src/com/agentnet/CreateAgentAction.java": java_action
        }
    
    def get_supported_ides(self) -> List[str]:
        """Get list of supported IDEs."""
        return self.supported_ides.copy()
    
    def generate_extension_for_ide(self, ide: str, extension_name: str) -> Dict[str, str]:
        """Generate extension for specified IDE."""
        if ide not in self.supported_ides:
            raise ValueError(f"IDE {ide} not supported. Supported IDEs: {self.supported_ides}")
        
        if ide == "vscode":
            return self.generate_vscode_extension(extension_name)
        elif ide == "jetbrains":
            return self.generate_jetbrains_plugin(extension_name)
        else:
            # For vim/emacs, return basic configuration files
            return {
                f"{ide}_config": f"# {ide.upper()} configuration for AgentNet\n# TODO: Implement {ide} integration"
            }