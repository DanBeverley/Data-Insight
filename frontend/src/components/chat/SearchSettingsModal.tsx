import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

export type SearchProvider = "duckduckgo" | "brave" | "searxng";

export interface SearchSettings {
    provider: SearchProvider;
    braveApiKey?: string;
    searxngUrl?: string;
}

interface SearchSettingsModalProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    settings: SearchSettings;
    onSave: (settings: SearchSettings) => void;
}

const STORAGE_KEY = "datainsight_search_settings";

export function loadSearchSettings(): SearchSettings {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) return JSON.parse(stored);
    } catch { }
    return { provider: "duckduckgo" };
}

export function saveSearchSettings(settings: SearchSettings): void {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

export function SearchSettingsModal({ open, onOpenChange, settings, onSave }: SearchSettingsModalProps) {
    const [provider, setProvider] = useState<SearchProvider>(settings.provider);
    const [braveApiKey, setBraveApiKey] = useState(settings.braveApiKey || "");
    const [searxngUrl, setSearxngUrl] = useState(settings.searxngUrl || "");

    useEffect(() => {
        if (open) {
            setProvider(settings.provider);
            setBraveApiKey(settings.braveApiKey || "");
            setSearxngUrl(settings.searxngUrl || "");
        }
    }, [open, settings]);

    const handleSave = () => {
        const newSettings: SearchSettings = {
            provider,
            ...(provider === "brave" && { braveApiKey }),
            ...(provider === "searxng" && { searxngUrl }),
        };
        saveSearchSettings(newSettings);
        onSave(newSettings);
        onOpenChange(false);
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-md bg-card border-border">
                <DialogHeader>
                    <DialogTitle className="text-foreground">Search Settings</DialogTitle>
                </DialogHeader>

                <div className="space-y-4 py-4">
                    <Label className="text-muted-foreground text-sm">Choose your preferred search provider:</Label>

                    <RadioGroup value={provider} onValueChange={(v) => setProvider(v as SearchProvider)} className="space-y-2">
                        <label className="flex items-center gap-3 px-4 py-3 rounded-lg border border-border bg-background hover:bg-muted/50 cursor-pointer transition-colors">
                            <RadioGroupItem value="searxng" id="searxng" />
                            <span className="text-foreground">SearXNG (Self-hosted)</span>
                        </label>
                        <label className="flex items-center gap-3 px-4 py-3 rounded-lg border border-border bg-background hover:bg-muted/50 cursor-pointer transition-colors">
                            <RadioGroupItem value="brave" id="brave" />
                            <span className="text-foreground">Brave Search</span>
                        </label>
                        <label className="flex items-center gap-3 px-4 py-3 rounded-lg border border-border bg-background hover:bg-muted/50 cursor-pointer transition-colors">
                            <RadioGroupItem value="duckduckgo" id="duckduckgo" />
                            <span className="text-foreground">DuckDuckGo</span>
                        </label>
                    </RadioGroup>

                    {provider === "brave" && (
                        <div className="space-y-2 pt-2">
                            <Label htmlFor="brave-key" className="text-muted-foreground text-sm">Brave API Key</Label>
                            <Input
                                id="brave-key"
                                type="password"
                                placeholder="Enter Brave API Key"
                                value={braveApiKey}
                                onChange={(e) => setBraveApiKey(e.target.value)}
                                className="bg-background border-border"
                            />
                            <p className="text-xs text-muted-foreground">API key required for Brave Search</p>
                        </div>
                    )}

                    {provider === "searxng" && (
                        <div className="space-y-2 pt-2">
                            <Label htmlFor="searxng-url" className="text-muted-foreground text-sm">SearXNG Instance URL</Label>
                            <Input
                                id="searxng-url"
                                type="url"
                                placeholder="https://your-searxng-instance.com"
                                value={searxngUrl}
                                onChange={(e) => setSearxngUrl(e.target.value)}
                                className="bg-background border-border"
                            />
                        </div>
                    )}
                </div>

                <DialogFooter className="gap-2">
                    <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
                    <Button onClick={handleSave} className="bg-primary text-primary-foreground">Save</Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
