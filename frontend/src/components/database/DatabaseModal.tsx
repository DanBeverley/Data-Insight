import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { X, Database, Loader2, CheckCircle, AlertCircle, Table, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "@/hooks/use-toast";

const STORAGE_KEY = "datainsight_db_credentials";

interface DatabaseModalProps {
    isOpen: boolean;
    onClose: () => void;
    sessionId: string;
    onDataLoaded?: (tableName: string, rowCount: number) => void;
}

type DbType = "postgresql" | "mysql" | "sqlite";

interface TableInfo {
    name: string;
    row_count: number;
}

export function DatabaseModal({ isOpen, onClose, sessionId, onDataLoaded }: DatabaseModalProps) {
    const [dbType, setDbType] = useState<DbType>("postgresql");
    const [host, setHost] = useState("");
    const [port, setPort] = useState("");
    const [database, setDatabase] = useState("");
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [filePath, setFilePath] = useState("");

    const [status, setStatus] = useState<"idle" | "testing" | "tested" | "connected" | "loading" | "error">("idle");
    const [loadingTable, setLoadingTable] = useState(false);
    const [errorMsg, setErrorMsg] = useState("");
    const [connectionId, setConnectionId] = useState<string | null>(null);
    const [tables, setTables] = useState<TableInfo[]>([]);
    const [selectedTables, setSelectedTables] = useState<Set<string>>(new Set());
    const [loadedTables, setLoadedTables] = useState<Set<string>>(new Set());
    const [isClosing, setIsClosing] = useState(false);
    const [refreshing, setRefreshing] = useState(false);

    const handleClose = useCallback(() => {
        setIsClosing(true);
        setTimeout(() => {
            setIsClosing(false);
            onClose();
        }, 150);
    }, [onClose]);

    const dbTypes: { id: DbType; label: string; icon: string }[] = [
        { id: "postgresql", label: "PostgreSQL", icon: "🐘" },
        { id: "mysql", label: "MySQL", icon: "🐬" },
        { id: "sqlite", label: "SQLite", icon: "📁" },
    ];

    const defaultPorts: Record<DbType, string> = {
        postgresql: "5432",
        mysql: "3306",
        sqlite: "",
    };

    // Load saved credentials on mount
    useEffect(() => {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {
                const creds = JSON.parse(saved);
                if (creds.dbType) setDbType(creds.dbType);
                if (creds.host) setHost(creds.host);
                if (creds.port) setPort(creds.port);
                if (creds.database) setDatabase(creds.database);
                if (creds.username) setUsername(creds.username);
                if (creds.password) setPassword(creds.password);
                if (creds.filePath) setFilePath(creds.filePath);
            }
        } catch (e) {
            console.warn("Failed to load saved credentials");
        }
    }, []);

    // Auto-save credentials when they change
    const saveCredentials = useCallback(() => {
        try {
            const creds = { dbType, host, port, database, username, password, filePath };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(creds));
        } catch (e) {
            console.warn("Failed to save credentials");
        }
    }, [dbType, host, port, database, username, password, filePath]);

    useEffect(() => {
        saveCredentials();
    }, [saveCredentials]);

    useEffect(() => {
        if (dbType !== "sqlite") {
            setPort(defaultPorts[dbType]);
        }
    }, [dbType]);

    // Reset session-specific state when sessionId changes
    useEffect(() => {
        setLoadedTables(new Set());
        setConnectionId(null);
        setTables([]);
        setSelectedTables(new Set());
        setStatus("idle");
    }, [sessionId]);

    const handleTest = async () => {
        setStatus("testing");
        setErrorMsg("");

        try {
            const response = await fetch("/api/connections/test", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    db_type: dbType,
                    host: dbType !== "sqlite" ? host : undefined,
                    port: dbType !== "sqlite" ? parseInt(port) : undefined,
                    database: dbType === "sqlite" ? filePath : database,
                    username: dbType !== "sqlite" ? username : undefined,
                    password: dbType !== "sqlite" ? password : undefined,
                    file_path: dbType === "sqlite" ? filePath : undefined,
                }),
            });

            const data = await response.json();

            if (data.success) {
                setStatus("tested");
                setTables(data.tables || []);
            } else {
                setStatus("error");
                setErrorMsg(data.error || data.message);
            }
        } catch (e) {
            setStatus("error");
            setErrorMsg("Connection failed");
        }
    };

    const handleConnect = async () => {
        setStatus("loading");

        try {
            const response = await fetch("/api/connections/connect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    db_type: dbType,
                    host: dbType !== "sqlite" ? host : undefined,
                    port: dbType !== "sqlite" ? parseInt(port) : undefined,
                    database: dbType === "sqlite" ? filePath : database,
                    username: dbType !== "sqlite" ? username : undefined,
                    password: dbType !== "sqlite" ? password : undefined,
                    file_path: dbType === "sqlite" ? filePath : undefined,
                }),
            });

            const data = await response.json();

            if (data.success || data.connection_id) {
                setConnectionId(data.connection_id);
                setTables(data.tables || []);
                setStatus("connected");
            } else {
                setStatus("error");
                setErrorMsg(data.detail || "Connection failed");
            }
        } catch (e) {
            setStatus("error");
            setErrorMsg("Connection failed");
        }
    };

    const handleRefresh = async () => {
        if (!connectionId) return;

        setRefreshing(true);
        try {
            const response = await fetch(`/api/connections/${connectionId}/tables`);
            const data = await response.json();

            if (data.tables) {
                setTables(data.tables);
                toast({
                    title: "Tables Refreshed",
                    description: `Found ${data.tables.length} table(s)`,
                });
            }
        } catch (e) {
            toast({
                title: "Refresh Failed",
                description: "Could not refresh tables",
                variant: "destructive",
            });
        } finally {
            setRefreshing(false);
        }
    };

    const toggleTableSelection = (tableName: string) => {
        setSelectedTables(prev => {
            const next = new Set(prev);
            if (next.has(tableName)) {
                next.delete(tableName);
            } else {
                next.add(tableName);
            }
            return next;
        });
    };

    const handleLoadTable = async () => {
        if (!connectionId || selectedTables.size === 0) return;

        setLoadingTable(true);
        setErrorMsg("");
        const tablesToLoad = Array.from(selectedTables).filter(t => !loadedTables.has(t));
        let successCount = 0;

        for (const tableName of tablesToLoad) {
            try {
                const response = await fetch(`/api/connections/${connectionId}/load`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        table_name: tableName,
                        limit: 50000,
                        session_id: sessionId,
                    }),
                });

                const data = await response.json();

                if (data.success) {
                    onDataLoaded?.(tableName, data.rows);
                    setLoadedTables(prev => new Set(prev).add(tableName));
                    successCount++;
                }
            } catch (e) {
                console.error(`Failed to load ${tableName}:`, e);
            }
        }

        if (successCount > 0) {
            toast({
                title: "Tables Loaded",
                description: `Successfully loaded ${successCount} table${successCount > 1 ? 's' : ''}`,
            });
        }
        setSelectedTables(new Set());
        setLoadingTable(false);
    };

    if (!isOpen && !isClosing) return null;

    return (
        <div className={cn(
            "fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm transition-opacity duration-150",
            isClosing ? "opacity-0" : "opacity-100"
        )}>
            <div className={cn(
                "w-[480px] bg-card border border-white/10 rounded-xl shadow-2xl overflow-hidden transition-all duration-150",
                isClosing
                    ? "animate-out fade-out zoom-out-95"
                    : "animate-in fade-in zoom-in-95"
            )}>
                <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
                    <div className="flex items-center gap-2">
                        <Database className="h-5 w-5 text-primary" />
                        <span className="font-semibold">Connect Database</span>
                    </div>
                    <Button variant="ghost" size="icon" onClick={handleClose} className="h-8 w-8">
                        <X className="h-4 w-4" />
                    </Button>
                </div>

                <div className="p-5 space-y-5">
                    <div className="flex gap-2">
                        {dbTypes.map((db) => (
                            <button
                                key={db.id}
                                onClick={() => setDbType(db.id)}
                                className={cn(
                                    "flex-1 py-2.5 px-3 rounded-lg border text-sm font-medium transition-all duration-200",
                                    dbType === db.id
                                        ? "border-primary bg-primary/10 text-primary"
                                        : "border-white/10 hover:border-white/20 hover:bg-white/5 text-muted-foreground"
                                )}
                            >
                                <span className="mr-1.5">{db.icon}</span>
                                {db.label}
                            </button>
                        ))}
                    </div>

                    {dbType === "sqlite" ? (
                        <div className="space-y-2 animate-in fade-in slide-in-from-bottom-2 duration-200">
                            <Label className="text-xs text-muted-foreground">Database File Path</Label>
                            <Input
                                value={filePath}
                                onChange={(e) => setFilePath(e.target.value)}
                                placeholder="/path/to/database.db"
                                className="bg-background/50 transition-all duration-200 focus:ring-2 focus:ring-primary/20"
                            />
                        </div>
                    ) : (
                        <div className="space-y-3 animate-in fade-in slide-in-from-bottom-2 duration-200">
                            <div className="grid grid-cols-3 gap-3">
                                <div className="col-span-2 space-y-1.5">
                                    <Label className="text-xs text-muted-foreground">Host</Label>
                                    <Input value={host} onChange={(e) => setHost(e.target.value)} placeholder="localhost" className="bg-background/50 transition-all duration-200 focus:ring-2 focus:ring-primary/20" />
                                </div>
                                <div className="space-y-1.5">
                                    <Label className="text-xs text-muted-foreground">Port</Label>
                                    <Input value={port} onChange={(e) => setPort(e.target.value)} className="bg-background/50 transition-all duration-200 focus:ring-2 focus:ring-primary/20" />
                                </div>
                            </div>
                            <div className="space-y-1.5">
                                <Label className="text-xs text-muted-foreground">Database</Label>
                                <Input value={database} onChange={(e) => setDatabase(e.target.value)} placeholder="my_database" className="bg-background/50 transition-all duration-200 focus:ring-2 focus:ring-primary/20" />
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="space-y-1.5">
                                    <Label className="text-xs text-muted-foreground">Username</Label>
                                    <Input value={username} onChange={(e) => setUsername(e.target.value)} className="bg-background/50 transition-all duration-200 focus:ring-2 focus:ring-primary/20" />
                                </div>
                                <div className="space-y-1.5">
                                    <Label className="text-xs text-muted-foreground">Password</Label>
                                    <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="bg-background/50 transition-all duration-200 focus:ring-2 focus:ring-primary/20" />
                                </div>
                            </div>
                        </div>
                    )}

                    {status === "error" && (
                        <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400 animate-in fade-in slide-in-from-top-2 duration-200">
                            <AlertCircle className="h-4 w-4 shrink-0" />
                            <span>{errorMsg}</span>
                        </div>
                    )}

                    {(status === "tested" || status === "connected") && tables.length > 0 && (
                        <div className="space-y-2 animate-in fade-in slide-in-from-bottom-3 duration-300">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <Label className="text-xs text-muted-foreground">Select Table</Label>
                                    <button
                                        onClick={handleRefresh}
                                        disabled={refreshing}
                                        className="p-1 rounded hover:bg-white/10 transition-colors disabled:opacity-50"
                                        title="Refresh tables"
                                    >
                                        <RefreshCw className={cn("h-3.5 w-3.5 text-muted-foreground", refreshing && "animate-spin")} />
                                    </button>
                                </div>
                                {loadedTables.size > 0 && (
                                    <span className="text-xs text-primary">
                                        {loadedTables.size} table{loadedTables.size > 1 ? 's' : ''} loaded
                                    </span>
                                )}
                            </div>
                            <div className="max-h-40 overflow-y-auto space-y-1 bg-background/30 rounded-lg p-2 border border-white/5">
                                {tables.map((table) => (
                                    <button
                                        key={table.name}
                                        onClick={() => !loadedTables.has(table.name) && toggleTableSelection(table.name)}
                                        className={cn(
                                            "w-full flex items-center justify-between px-3 py-2 rounded text-sm transition-all duration-150",
                                            loadedTables.has(table.name)
                                                ? "bg-primary/10 text-primary/70 cursor-default"
                                                : selectedTables.has(table.name)
                                                    ? "bg-primary/20 text-primary"
                                                    : "hover:bg-white/5 text-muted-foreground"
                                        )}
                                    >
                                        <span className="flex items-center gap-2">
                                            {loadedTables.has(table.name) ? (
                                                <CheckCircle className="h-3.5 w-3.5 text-primary" />
                                            ) : (
                                                <div className={cn(
                                                    "h-3.5 w-3.5 rounded-full border-2 flex items-center justify-center transition-colors",
                                                    selectedTables.has(table.name)
                                                        ? "border-primary bg-primary"
                                                        : "border-muted-foreground/50"
                                                )}>
                                                    {selectedTables.has(table.name) && (
                                                        <div className="h-1.5 w-1.5 rounded-full bg-background" />
                                                    )}
                                                </div>
                                            )}
                                            {table.name}
                                        </span>
                                        <span className="text-xs opacity-60">
                                            {loadedTables.has(table.name) ? "Loaded" : `${table.row_count?.toLocaleString()} rows`}
                                        </span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                <div className="flex gap-2 px-5 py-4 border-t border-white/10 bg-white/[0.02]">
                    {status !== "connected" ? (
                        <>
                            <Button
                                variant="outline"
                                onClick={handleTest}
                                disabled={status === "testing" || status === "loading"}
                                className="flex-1 transition-all duration-200 hover:bg-white/10 hover:border-primary/50 active:scale-[0.98]"
                            >
                                {status === "testing" ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                                {status === "tested" ? "✓ Tested" : "Test"}
                            </Button>
                            <Button
                                onClick={handleConnect}
                                disabled={status === "testing" || status === "loading"}
                                className="flex-1 transition-all duration-200 hover:brightness-110 active:scale-[0.98]"
                            >
                                {status === "loading" ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                                Connect
                            </Button>
                        </>
                    ) : (
                        <>
                            <Button
                                variant="outline"
                                onClick={handleClose}
                                className="transition-all duration-200 hover:bg-white/10 active:scale-[0.98]"
                            >
                                {loadedTables.size > 0 ? "Done" : "Cancel"}
                            </Button>
                            <Button
                                onClick={handleLoadTable}
                                disabled={selectedTables.size === 0 || loadingTable}
                                className="flex-1 transition-all duration-200 hover:brightness-110 active:scale-[0.98] disabled:opacity-50"
                            >
                                {loadingTable ? (
                                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                                ) : (
                                    <CheckCircle className="h-4 w-4 mr-2" />
                                )}
                                {selectedTables.size === 0 ? "Select Tables" : `Load ${selectedTables.size} Table${selectedTables.size > 1 ? 's' : ''}`}
                            </Button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
