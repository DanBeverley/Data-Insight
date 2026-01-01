import { useEffect, useCallback, useRef } from 'react';
import { toast, Toaster } from 'sonner';

interface Notification {
    id: string;
    title: string;
    message: string;
    type: 'info' | 'success' | 'warning' | 'error' | 'alert';
    created_at: string;
}

export function useNotifications(pollInterval = 30000) {
    const lastCheckRef = useRef<string | null>(null);

    const fetchPending = useCallback(async () => {
        try {
            const response = await fetch('/api/notifications/pending');
            if (!response.ok) return;

            const notifications: Notification[] = await response.json();

            notifications.forEach((notification) => {
                if (lastCheckRef.current && notification.created_at <= lastCheckRef.current) return;

                switch (notification.type) {
                    case 'success':
                        toast.success(notification.title, { description: notification.message });
                        break;
                    case 'error':
                        toast.error(notification.title, { description: notification.message });
                        break;
                    case 'warning':
                    case 'alert':
                        toast.warning(notification.title, { description: notification.message });
                        break;
                    default:
                        toast.info(notification.title, { description: notification.message });
                }
            });

            if (notifications.length > 0) {
                lastCheckRef.current = notifications[0].created_at;
            }
        } catch (error) {
            console.error('Failed to fetch notifications:', error);
        }
    }, []);

    useEffect(() => {
        fetchPending();
        const interval = setInterval(fetchPending, pollInterval);
        return () => clearInterval(interval);
    }, [fetchPending, pollInterval]);

    return { fetchPending };
}

export function NotificationToaster() {
    return (
        <Toaster
            position="top-right"
            toastOptions={{
                style: {
                    background: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    color: 'hsl(var(--card-foreground))',
                },
                className: 'shadow-lg',
            }}
            richColors
            closeButton
        />
    );
}
