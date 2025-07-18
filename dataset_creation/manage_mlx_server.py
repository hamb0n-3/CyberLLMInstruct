#!/usr/bin/env python3
"""
MLX Server Management Script

This script helps manage the MLX advanced server for the dataset creation pipeline.
It can start, stop, check status, and monitor the server.
"""

import argparse
import subprocess
import psutil
import requests
import time
import json
import yaml
import logging
import signal
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLXServerManager:
    """Manages the MLX advanced server lifecycle"""
    
    def __init__(self, config_path: str = "pipeline_config.yaml"):
        self.config = self._load_config(config_path)
        self.server_url = self.config["mlx_server"]["base_url"]
        self.server_process = None
        self.pid_file = Path(".mlx_server.pid")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            # Return default configuration
            return {
                "mlx_server": {"base_url": "http://localhost:8080"},
                "advanced_server": {
                    "model": "mlx-community/c4ai-command-r-v01-4bit",
                    "draft_model": "mlx-community/Phi-3-mini-4k-instruct-4bit",
                    "max_batch_size": 16,
                    "batch_timeout_ms": 50,
                    "use_combined": True
                }
            }
    
    def start_server(self, 
                    model: Optional[str] = None,
                    draft_model: Optional[str] = None,
                    port: Optional[int] = None,
                    use_combined: Optional[bool] = None,
                    detach: bool = False) -> bool:
        """Start the MLX advanced server"""
        
        # Check if server is already running
        if self.is_running():
            logger.warning("Server is already running")
            return True
        
        # Get configuration
        server_config = self.config["advanced_server"]
        model = model or server_config["model"]
        draft_model = draft_model or server_config["draft_model"]
        use_combined = use_combined if use_combined is not None else server_config.get("use_combined", True)
        
        # Extract port from URL if not specified
        if port is None:
            import urllib.parse
            parsed = urllib.parse.urlparse(self.server_url)
            port = parsed.port or 8080
        
        # Build command
        cmd = [
            sys.executable, "-m", "mlx_parallm.advanced_server",
            "--model", model,
            "--draft-model", draft_model,
            "--port", str(port),
            "--max-batch-size", str(server_config["max_batch_size"]),
            "--batch-timeout-ms", str(server_config["batch_timeout_ms"])
        ]
        
        if use_combined:
            cmd.append("--use-combined")
        
        logger.info(f"Starting MLX server with command: {' '.join(cmd)}")
        
        try:
            if detach:
                # Start in background
                # Create log file for server output
                log_file = open('.mlx_server.log', 'w')
                self.server_process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                    start_new_session=True
                )
                
                # Save PID
                with open(self.pid_file, 'w') as f:
                    f.write(str(self.server_process.pid))
                
                # Wait for server to start
                logger.info("Waiting for server to start...")
                if self._wait_for_server(timeout=30):
                    logger.info(f"Server started successfully (PID: {self.server_process.pid})")
                    log_file.close()
                    return True
                else:
                    logger.error("Server failed to start within timeout")
                    # Try to get error from log
                    log_file.close()
                    try:
                        with open('.mlx_server.log', 'r') as f:
                            error_output = f.read()
                            if error_output:
                                logger.error(f"Server error output:\n{error_output}")
                    except:
                        pass
                    self.stop_server()
                    return False
            else:
                # Run in foreground
                subprocess.run(cmd)
                return True
                
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """Stop the MLX server"""
        # Try to get PID from file first
        if self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process exists
                try:
                    process = psutil.Process(pid)
                    if "mlx_parallm" in " ".join(process.cmdline()):
                        logger.info(f"Stopping server (PID: {pid})")
                        process.terminate()
                        
                        # Wait for graceful shutdown
                        try:
                            process.wait(timeout=10)
                        except psutil.TimeoutExpired:
                            logger.warning("Server didn't stop gracefully, forcing...")
                            process.kill()
                        
                        # Remove PID file
                        self.pid_file.unlink()
                        logger.info("Server stopped successfully")
                        return True
                except psutil.NoSuchProcess:
                    logger.warning("Server process not found")
                    self.pid_file.unlink()
                    
            except Exception as e:
                logger.error(f"Error reading PID file: {e}")
        
        # Fallback: Find process by name
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = " ".join(proc.info['cmdline'] or [])
                if "mlx_parallm.advanced_server" in cmdline:
                    logger.info(f"Found server process (PID: {proc.info['pid']})")
                    proc.terminate()
                    proc.wait(timeout=10)
                    logger.info("Server stopped successfully")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        logger.warning("No MLX server process found")
        return False
    
    def restart_server(self, **kwargs) -> bool:
        """Restart the MLX server"""
        logger.info("Restarting server...")
        self.stop_server()
        time.sleep(2)  # Give it time to fully stop
        return self.start_server(**kwargs)
    
    def is_running(self) -> bool:
        """Check if the server is running"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _wait_for_server(self, timeout: int = 30) -> bool:
        """Wait for server to become available"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_running():
                return True
            time.sleep(1)
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status and statistics"""
        status = {
            "running": self.is_running(),
            "url": self.server_url
        }
        
        if status["running"]:
            try:
                # Get server stats
                response = requests.get(f"{self.server_url}/v1/stats", timeout=5)
                if response.status_code == 200:
                    status["stats"] = response.json()
                
                # Get health info
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    status["health"] = response.json()
                    
            except Exception as e:
                logger.error(f"Error getting server stats: {e}")
        
        # Check process info
        if self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                process = psutil.Process(pid)
                status["process"] = {
                    "pid": pid,
                    "cpu_percent": process.cpu_percent(interval=1),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "create_time": process.create_time()
                }
            except:
                pass
        
        return status
    
    def monitor(self, interval: int = 5):
        """Monitor server status continuously"""
        logger.info(f"Monitoring server at {self.server_url} (Ctrl+C to stop)")
        
        try:
            while True:
                status = self.get_status()
                
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Display status
                print(f"MLX Server Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 60)
                print(f"Status: {'RUNNING' if status['running'] else 'STOPPED'}")
                print(f"URL: {status['url']}")
                
                if status.get("process"):
                    proc = status["process"]
                    print(f"\nProcess Info:")
                    print(f"  PID: {proc['pid']}")
                    print(f"  CPU: {proc['cpu_percent']:.1f}%")
                    print(f"  Memory: {proc['memory_mb']:.1f} MB")
                    print(f"  Uptime: {int(time.time() - proc['create_time'])} seconds")
                
                if status.get("stats"):
                    stats = status["stats"]
                    print(f"\nServer Statistics:")
                    print(f"  Mode: {stats.get('mode', 'unknown')}")
                    print(f"  Total Requests: {stats.get('total_requests', 0)}")
                    print(f"  Avg Latency: {stats.get('average_latency', 0):.3f}s")
                    
                    if stats.get("combined_stats"):
                        cs = stats["combined_stats"]
                        print(f"\nCombined Mode Stats:")
                        print(f"  Avg Batch Size: {cs.get('avg_batch_size', 0):.2f}")
                        print(f"  Acceptance Rate: {cs.get('avg_acceptance_rate', 0):.2%}")
                        if cs.get('speculative_speedup'):
                            print(f"  Speculative Speedup: {cs['speculative_speedup']:.2f}x")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")


def main():
    parser = argparse.ArgumentParser(description="Manage MLX Advanced Server")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the server')
    start_parser.add_argument('--model', help='Target model path')
    start_parser.add_argument('--draft-model', help='Draft model path')
    start_parser.add_argument('--port', type=int, help='Server port')
    start_parser.add_argument('--use-combined', action='store_true', help='Use combined mode')
    start_parser.add_argument('--no-combined', dest='use_combined', action='store_false', help='Disable combined mode')
    start_parser.add_argument('--detach', '-d', action='store_true', help='Run in background')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop the server')
    
    # Restart command
    restart_parser = subparsers.add_parser('restart', help='Restart the server')
    restart_parser.add_argument('--model', help='Target model path')
    restart_parser.add_argument('--draft-model', help='Draft model path')
    restart_parser.add_argument('--port', type=int, help='Server port')
    restart_parser.add_argument('--detach', '-d', action='store_true', help='Run in background')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check server status')
    status_parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor server continuously')
    monitor_parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    
    # Config file
    parser.add_argument('--config', default='pipeline_config.yaml', help='Pipeline configuration file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create manager
    manager = MLXServerManager(args.config)
    
    # Execute command
    if args.command == 'start':
        success = manager.start_server(
            model=args.model,
            draft_model=args.draft_model,
            port=args.port,
            use_combined=args.use_combined,
            detach=args.detach
        )
        sys.exit(0 if success else 1)
        
    elif args.command == 'stop':
        success = manager.stop_server()
        sys.exit(0 if success else 1)
        
    elif args.command == 'restart':
        success = manager.restart_server(
            model=args.model,
            draft_model=args.draft_model,
            port=args.port,
            detach=args.detach
        )
        sys.exit(0 if success else 1)
        
    elif args.command == 'status':
        status = manager.get_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Server Status: {'RUNNING' if status['running'] else 'STOPPED'}")
            print(f"URL: {status['url']}")
            if status.get('stats'):
                print(f"Mode: {status['stats'].get('mode', 'unknown')}")
                print(f"Total Requests: {status['stats'].get('total_requests', 0)}")
        
    elif args.command == 'monitor':
        manager.monitor(interval=args.interval)


if __name__ == "__main__":
    main()