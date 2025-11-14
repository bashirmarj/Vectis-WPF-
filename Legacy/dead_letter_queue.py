"""
Dead Letter Queue for failed CAD analysis requests
Stores failures with context for later analysis and reprocessing
"""

import os
import json
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Supabase connection
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None


class DeadLetterQueue:
    """
    Stores failed requests with complete error context.
    
    Enables:
    - Later analysis of failure patterns
    - Manual reprocessing after fixes
    - Tracking retry counts
    - Error pattern detection
    """
    
    def __init__(self, table_name: str = 'failed_cad_analyses'):
        self.table_name = table_name
        
        if not supabase:
            logger.warning("Dead letter queue: Supabase not configured, using local logging only")
    
    def store_failure(
        self,
        correlation_id: str,
        file_path: str,
        error_type: str,  # 'transient', 'permanent', 'systemic'
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> bool:
        """
        Store failed request in dead letter queue.
        
        Args:
            correlation_id: Request correlation ID for distributed tracing
            file_path: Path to the CAD file that failed
            error_type: 'transient', 'permanent', or 'systemic'
            error_message: Short error message
            error_details: Additional error context (exception, traceback, etc.)
            retry_count: Number of retry attempts made
            
        Returns:
            True if stored successfully, False otherwise
        """
        
        failure_record = {
            'correlation_id': correlation_id,
            'file_path': file_path,
            'error_type': error_type,
            'error_message': error_message,
            'error_details': error_details or {},
            'retry_count': retry_count,
            'traceback': traceback.format_exc() if error_details and error_details.get('include_traceback') else None,
            'created_at': datetime.utcnow().isoformat(),
        }
        
        try:
            if supabase:
                # Store in database
                result = supabase.table(self.table_name).insert(failure_record).execute()
                logger.info(f"📝 Dead letter queue: Stored failure {correlation_id} ({error_type})")
                return True
            else:
                # Fallback to file system logging
                self._store_to_file(failure_record)
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to store in dead letter queue: {str(e)}")
            # Still log to file as fallback
            self._store_to_file(failure_record)
            return False
    
    def _store_to_file(self, record: Dict[str, Any]):
        """Fallback: Store failure record to local file"""
        try:
            dlq_dir = '/tmp/dead_letter_queue'
            os.makedirs(dlq_dir, exist_ok=True)
            
            filename = f"{dlq_dir}/{record['correlation_id']}.json"
            with open(filename, 'w') as f:
                json.dump(record, f, indent=2)
            
            logger.info(f"📝 Dead letter queue: Stored to file {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to store to file: {str(e)}")
    
    def get_failures(
        self,
        error_type: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        Retrieve failed requests for analysis.
        
        Args:
            error_type: Filter by error type (optional)
            limit: Maximum number of records to retrieve
            
        Returns:
            List of failure records
        """
        
        if not supabase:
            logger.warning("Cannot retrieve failures: Supabase not configured")
            return []
        
        try:
            query = supabase.table(self.table_name).select('*').order('created_at', desc=True).limit(limit)
            
            if error_type:
                query = query.eq('error_type', error_type)
            
            result = query.execute()
            return result.data
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve failures: {str(e)}")
            return []
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about failures for monitoring.
        
        Returns:
            Dictionary with failure counts by type, retry distribution, etc.
        """
        
        if not supabase:
            return {'error': 'Supabase not configured'}
        
        try:
            # Get all failures from last 24 hours
            from datetime import timedelta
            cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            
            result = supabase.table(self.table_name).select('error_type, retry_count').gte('created_at', cutoff).execute()
            
            failures = result.data
            
            stats = {
                'total_failures_24h': len(failures),
                'by_error_type': {
                    'transient': sum(1 for f in failures if f['error_type'] == 'transient'),
                    'permanent': sum(1 for f in failures if f['error_type'] == 'permanent'),
                    'systemic': sum(1 for f in failures if f['error_type'] == 'systemic'),
                },
                'avg_retry_count': sum(f['retry_count'] for f in failures) / len(failures) if failures else 0,
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {str(e)}")
            return {'error': str(e)}


# Global dead letter queue instance
dlq = DeadLetterQueue()
