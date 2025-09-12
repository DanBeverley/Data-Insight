-- Production-grade SQL queries for meta-learning database operations

-- Get system learning summary with optimized aggregations
CREATE OR REPLACE VIEW system_learning_summary AS
SELECT 
    COUNT(*) as total_executions,
    COUNT(*) FILTER (WHERE success_rating >= 0.8) as high_success_executions,
    ROUND(AVG(success_rating) FILTER (WHERE success_rating > 0), 4) as avg_success_rating,
    COUNT(DISTINCT dc.domain) as domains_covered,
    COUNT(DISTINCT lp.pattern_id) as patterns_discovered,
    ROUND(AVG(success_rating) FILTER (WHERE timestamp > NOW() - INTERVAL '30 days' AND success_rating > 0), 4) as recent_avg_success,
    CASE 
        WHEN AVG(success_rating) FILTER (WHERE timestamp > NOW() - INTERVAL '30 days' AND success_rating > 0) > 
             AVG(success_rating) FILTER (WHERE success_rating > 0) 
        THEN 'improving' 
        ELSE 'stable' 
    END as performance_trend,
    ROUND(LEAST(1.0, COUNT(*) / 1000.0), 2) as meta_learning_maturity,
    NOW() as last_updated
FROM pipeline_executions pe
LEFT JOIN dataset_characteristics dc ON pe.dataset_hash = dc.dataset_hash
LEFT JOIN learning_patterns lp ON TRUE;

-- Find similar successful executions with similarity scoring
CREATE OR REPLACE FUNCTION find_similar_executions(
    target_samples INTEGER,
    target_features INTEGER, 
    target_domain TEXT,
    target_complexity REAL,
    min_success_rating REAL DEFAULT 0.7,
    result_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    execution_id TEXT,
    similarity_score REAL,
    success_rating REAL,
    execution_time REAL,
    config_hash TEXT,
    user_satisfaction REAL
) AS $$
BEGIN
    RETURN QUERY
    WITH similarity_calc AS (
        SELECT 
            pe.execution_id,
            pe.success_rating,
            pe.execution_time,
            pe.config_hash,
            pe.user_satisfaction,
            (
                -- Dataset size similarity (weight: 0.3)
                0.3 * (1.0 - ABS(dc.n_samples - target_samples)::REAL / GREATEST(dc.n_samples, target_samples)) +
                -- Feature count similarity (weight: 0.2) 
                0.2 * (1.0 - ABS(dc.n_features - target_features)::REAL / GREATEST(dc.n_features, target_features)) +
                -- Domain match (weight: 0.3)
                0.3 * CASE WHEN dc.domain = target_domain THEN 1.0 ELSE 0.0 END +
                -- Complexity similarity (weight: 0.2)
                0.2 * (1.0 - ABS(dc.task_complexity_score - target_complexity) / GREATEST(dc.task_complexity_score, target_complexity))
            ) as calc_similarity
        FROM pipeline_executions pe
        JOIN dataset_characteristics dc ON pe.dataset_hash = dc.dataset_hash
        WHERE pe.success_rating >= min_success_rating 
        AND pe.validation_success = TRUE
        AND (pe.user_satisfaction IS NULL OR pe.user_satisfaction >= 0.6)
    )
    SELECT 
        sc.execution_id,
        sc.calc_similarity,
        sc.success_rating,
        sc.execution_time,
        sc.config_hash,
        sc.user_satisfaction
    FROM similarity_calc sc
    WHERE sc.calc_similarity >= 0.6
    ORDER BY 
        CASE WHEN sc.user_satisfaction IS NOT NULL THEN sc.user_satisfaction * 0.4 + sc.calc_similarity * 0.6 
             ELSE sc.calc_similarity * 0.8 END DESC,
        sc.success_rating DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Get domain performance analytics
CREATE OR REPLACE VIEW domain_performance_analytics AS
SELECT 
    dc.domain,
    COUNT(*) as total_projects,
    ROUND(AVG(pe.success_rating), 3) as avg_success_rate,
    ROUND(AVG(pe.execution_time), 2) as avg_execution_time,
    ROUND(AVG(pe.trust_score), 3) as avg_trust_score,
    COUNT(*) FILTER (WHERE pe.validation_success = TRUE) as successful_validations,
    ROUND(AVG(pe.budget_compliance_rate), 3) as avg_budget_compliance,
    ROUND(STDDEV(pe.success_rating), 3) as performance_consistency
FROM dataset_characteristics dc
JOIN pipeline_executions pe ON dc.dataset_hash = pe.dataset_hash
GROUP BY dc.domain
HAVING COUNT(*) >= 3
ORDER BY avg_success_rate DESC;

-- Strategy effectiveness analysis
CREATE OR REPLACE VIEW strategy_effectiveness AS
SELECT 
    pc.strategy_applied,
    pc.objective,
    pc.domain,
    COUNT(*) as usage_count,
    ROUND(AVG(pe.success_rating), 3) as avg_performance,
    ROUND(AVG(pe.execution_time), 2) as avg_time,
    COUNT(*) FILTER (WHERE pe.success_rating >= 0.8) as high_performance_count,
    ROUND(COUNT(*) FILTER (WHERE pe.success_rating >= 0.8)::REAL / COUNT(*), 3) as success_rate
FROM project_configs pc
JOIN pipeline_executions pe ON pc.config_hash = pe.config_hash
GROUP BY pc.strategy_applied, pc.objective, pc.domain
HAVING COUNT(*) >= 2
ORDER BY success_rate DESC, avg_performance DESC;

-- Learning pattern discovery query
WITH pattern_candidates AS (
    SELECT 
        pc.objective,
        pc.domain,
        pc.feature_engineering_enabled,
        pc.feature_selection_enabled,
        dc.target_type,
        CASE 
            WHEN dc.n_features < 20 THEN 'low_dimensional'
            WHEN dc.n_features < 100 THEN 'medium_dimensional' 
            ELSE 'high_dimensional'
        END as dimensionality,
        COUNT(*) as pattern_frequency,
        ROUND(AVG(pe.success_rating), 3) as avg_success,
        ARRAY_AGG(pe.execution_id ORDER BY pe.success_rating DESC LIMIT 3) as top_executions
    FROM project_configs pc
    JOIN pipeline_executions pe ON pc.config_hash = pe.config_hash
    JOIN dataset_characteristics dc ON pe.dataset_hash = dc.dataset_hash
    WHERE pe.success_rating >= 0.75
    GROUP BY 1,2,3,4,5,6
    HAVING COUNT(*) >= 3 AND AVG(pe.success_rating) >= 0.8
)
INSERT INTO learning_patterns (
    pattern_id, 
    pattern_type, 
    dataset_context, 
    config_elements, 
    success_indicators,
    confidence_score,
    usage_count,
    improvement_evidence
)
SELECT 
    'pattern_' || generate_random_uuid() as pattern_id,
    'successful_config' as pattern_type,
    jsonb_build_object(
        'domain', domain,
        'target_type', target_type,
        'dimensionality', dimensionality
    ) as dataset_context,
    jsonb_build_object(
        'objective', objective,
        'feature_engineering', feature_engineering_enabled,
        'feature_selection', feature_selection_enabled
    ) as config_elements,
    jsonb_build_object(
        'avg_success_rating', avg_success,
        'pattern_frequency', pattern_frequency
    ) as success_indicators,
    LEAST(1.0, avg_success * (pattern_frequency / 10.0)) as confidence_score,
    pattern_frequency as usage_count,
    jsonb_build_array(top_executions) as improvement_evidence
FROM pattern_candidates
ON CONFLICT (pattern_id) DO UPDATE SET
    usage_count = EXCLUDED.usage_count,
    confidence_score = EXCLUDED.confidence_score,
    last_validated = NOW();

-- Performance monitoring indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_executions_performance 
ON pipeline_executions (success_rating DESC, timestamp DESC) 
WHERE success_rating > 0;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_executions_domain_time 
ON pipeline_executions (timestamp DESC) 
INCLUDE (success_rating, execution_time);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dataset_similarity 
ON dataset_characteristics (domain, n_samples, n_features, task_complexity_score);

-- Cleanup old data efficiently  
CREATE OR REPLACE FUNCTION cleanup_old_executions(retention_days INTEGER DEFAULT 365)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    cutoff_date TIMESTAMP;
BEGIN
    cutoff_date := NOW() - (retention_days || ' days')::INTERVAL;
    
    -- Delete in batches to avoid lock contention
    WITH old_executions AS (
        DELETE FROM execution_feedback 
        WHERE execution_id IN (
            SELECT execution_id FROM pipeline_executions 
            WHERE timestamp < cutoff_date 
            LIMIT 1000
        )
        RETURNING execution_id
    ),
    deleted_executions AS (
        DELETE FROM pipeline_executions 
        WHERE execution_id IN (SELECT execution_id FROM old_executions)
        RETURNING 1
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted_executions;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;