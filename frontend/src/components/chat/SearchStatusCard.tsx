import React from 'react';

interface SearchStatusItem {
    action: 'searching' | 'browsing' | 'complete';
    query?: string;
    provider?: string;
    url?: string;
    resultCount?: number;
}

interface SearchStatusCardProps {
    searchStatus: SearchStatusItem | null;
    searchHistory?: SearchStatusItem[];
}

export const SearchStatusCard: React.FC<SearchStatusCardProps> = ({ searchStatus, searchHistory = [] }) => {
    if (!searchStatus && searchHistory.length === 0) return null;

    const allItems = searchHistory.length > 0 ? searchHistory : (searchStatus ? [searchStatus] : []);

    return (
        <div className="search-status-inline" style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
            marginBottom: '12px',
            paddingLeft: '8px'
        }}>
            {allItems.map((item, index) => (
                <div key={index} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    fontSize: '14px',
                    color: 'rgba(255, 255, 255, 0.7)'
                }}>
                    {(item.action === 'searching' || item.action === 'complete') && (
                        <>
                            <span style={{ fontSize: '14px' }}>
                                {item.action === 'searching' ? '○' : '✓'}
                            </span>
                            <span style={{ color: 'rgba(255, 255, 255, 0.85)' }}>
                                Searching the web
                            </span>
                            <span style={{
                                color: item.action === 'complete' ? '#4ade80' : '#60a5fa',
                                fontWeight: 500
                            }}>
                                {item.resultCount || 0} results
                            </span>
                        </>
                    )}

                    {item.action === 'browsing' && item.url && (
                        <>
                            <span style={{ fontSize: '14px' }}>🌐</span>
                            <span style={{ color: 'rgba(255, 255, 255, 0.6)' }}>Browsing</span>
                            <span style={{
                                color: 'rgba(100, 160, 255, 0.8)',
                                whiteSpace: 'nowrap',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                maxWidth: '400px'
                            }}>
                                {item.url}
                            </span>
                        </>
                    )}
                </div>
            ))}

            <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-3px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .search-status-inline > div {
          animation: fadeIn 0.3s ease-in-out;
        }
      `}</style>
        </div>
    );
};

export default SearchStatusCard;
