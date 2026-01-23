import React from 'react';
import { Search, Globe } from 'lucide-react';

interface SearchStatusItem {
    action: 'searching' | 'browsing' | 'complete';
    query?: string;
    provider?: string;
    url?: string;
    resultCount?: number;
    sources?: string[];
}

interface SearchStatusCardProps {
    searchStatus: SearchStatusItem | null;
    searchHistory?: SearchStatusItem[];
}

export const SearchStatusCard: React.FC<SearchStatusCardProps> = ({ searchStatus, searchHistory = [] }) => {
    if (!searchStatus && searchHistory.length === 0) return null;

    // Get the latest search result count (from complete status or most recent with count)
    const latestComplete = [...searchHistory].reverse().find(item =>
        item.action === 'complete' || (item.resultCount && item.resultCount > 0)
    );
    const resultCount = latestComplete?.resultCount || searchStatus?.resultCount || 0;

    // Get current browsing URL (most recent browsing action)
    const currentBrowsing = searchStatus?.action === 'browsing' ? searchStatus :
        [...searchHistory].reverse().find(item => item.action === 'browsing');

    // Determine if still searching
    const isSearching = searchStatus?.action === 'searching';
    const isComplete = latestComplete?.action === 'complete' || resultCount > 0;

    return (
        <div className="flex flex-col gap-1.5 mb-3 pl-2">
            {/* Main search status line */}
            <div className="flex items-center gap-2 text-sm">
                <Search className={`w-4 h-4 ${isSearching ? 'animate-pulse text-primary' : 'text-muted-foreground'}`} />
                <span className="text-foreground/85">
                    {isSearching ? 'Searching the web' : 'Searched the web'}
                </span>
                <span className={`font-medium ${isComplete ? 'text-green-400' : 'text-primary'}`}>
                    {resultCount} results
                </span>
            </div>

            {/* Current browsing URL */}
            {currentBrowsing?.url && (
                <div className="flex items-center gap-2 text-sm animate-in fade-in slide-in-from-left-2 duration-200">
                    <Globe className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Browsing</span>
                    <span className="text-blue-400/80 truncate max-w-[400px]" title={currentBrowsing.url}>
                        {currentBrowsing.url}
                    </span>
                </div>
            )}
        </div>
    );
};

export default SearchStatusCard;

