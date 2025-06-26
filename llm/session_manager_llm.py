from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

class SessionManager:
    """Manages ADK sessions across multiple agents to prevent session loss."""
    
    def __init__(self):
        self.session_services = {}
        self.sessions = {}
        self.runners = {}
    
    def create_agent_session(self, agent_name, app_name, user_id, session_id):
        """Create a new session for an agent."""
        # Create unique session service for this agent
        session_service_key = f"{agent_name}_{app_name}"
        
        if session_service_key not in self.session_services:
            self.session_services[session_service_key] = InMemorySessionService()
        
        session_service = self.session_services[session_service_key]
        
        # Create session
        session = session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        
        # Store session reference
        session_key = f"{user_id}_{session_id}"
        self.sessions[session_key] = {
            'session': session,
            'service': session_service,
            'app_name': app_name,
            'user_id': user_id,
            'session_id': session_id
        }
        
        #print(f"Created session: {session_key} for {agent_name}")
        return session_service
    
    def create_runner(self, agent, runner_key, app_name):
        """Create and store a runner for an agent."""
        session_service_key = f"{runner_key}_{app_name}"
        session_service = self.session_services.get(session_service_key)
        
        if not session_service:
            raise ValueError(f"No session service found for {session_service_key}")
        
        runner = Runner(
            agent=agent,
            app_name=app_name,
            session_service=session_service,
        )
        
        self.runners[runner_key] = runner
        #print(f"Created runner: {runner_key}")
        return runner
    
    def get_session_info(self, user_id, session_id):
        """Get session information."""
        session_key = f"{user_id}_{session_id}"
        return self.sessions.get(session_key)
    
    def list_sessions(self):
        """List all active sessions for debugging."""
        print("\nActive Sessions:")
        for key, session_info in self.sessions.items():
            print(f"  {key}: {session_info['app_name']}")