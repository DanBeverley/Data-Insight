import { useEffect, useState, useCallback } from 'react';

interface PushSubscriptionState {
    isSupported: boolean;
    isSubscribed: boolean;
    permission: NotificationPermission | null;
}

function urlBase64ToUint8Array(base64String: string): Uint8Array {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    for (let i = 0; i < rawData.length; ++i) {
        outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
}

export function usePushNotifications() {
    const [state, setState] = useState<PushSubscriptionState>({
        isSupported: false,
        isSubscribed: false,
        permission: null
    });

    useEffect(() => {
        const isSupported = 'serviceWorker' in navigator && 'PushManager' in window;
        setState(prev => ({
            ...prev,
            isSupported,
            permission: isSupported ? Notification.permission : null
        }));

        if (isSupported && navigator.serviceWorker.controller) {
            checkSubscription();
        }
    }, []);

    const checkSubscription = async () => {
        try {
            const registration = await navigator.serviceWorker.ready;
            const subscription = await registration.pushManager.getSubscription();
            setState(prev => ({ ...prev, isSubscribed: !!subscription }));
        } catch (error) {
            console.error('Failed to check subscription:', error);
        }
    };

    const registerServiceWorker = async () => {
        if (!('serviceWorker' in navigator)) return null;

        try {
            const registration = await navigator.serviceWorker.register('/sw.js');
            await navigator.serviceWorker.ready;
            return registration;
        } catch (error) {
            console.error('Service worker registration failed:', error);
            return null;
        }
    };

    const requestPermission = useCallback(async () => {
        if (!state.isSupported) return false;

        const permission = await Notification.requestPermission();
        setState(prev => ({ ...prev, permission }));
        return permission === 'granted';
    }, [state.isSupported]);

    const subscribe = useCallback(async () => {
        if (!state.isSupported) return null;

        try {
            const registration = await registerServiceWorker();
            if (!registration) return null;

            const response = await fetch('/api/notifications/vapid-public-key');
            if (!response.ok) {
                console.error('VAPID key not configured');
                return null;
            }
            const { publicKey } = await response.json();

            const subscription = await registration.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: urlBase64ToUint8Array(publicKey) as BufferSource
            });

            await fetch('/api/notifications/subscribe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    endpoint: subscription.endpoint,
                    keys: {
                        p256dh: btoa(String.fromCharCode.apply(null, Array.from(new Uint8Array(subscription.getKey('p256dh')!)))),
                        auth: btoa(String.fromCharCode.apply(null, Array.from(new Uint8Array(subscription.getKey('auth')!))))
                    }
                })
            });

            setState(prev => ({ ...prev, isSubscribed: true }));
            return subscription;
        } catch (error) {
            console.error('Push subscription failed:', error);
            return null;
        }
    }, [state.isSupported]);

    const unsubscribe = useCallback(async () => {
        try {
            const registration = await navigator.serviceWorker.ready;
            const subscription = await registration.pushManager.getSubscription();

            if (subscription) {
                await subscription.unsubscribe();
                await fetch('/api/notifications/unsubscribe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ endpoint: subscription.endpoint })
                });
            }

            setState(prev => ({ ...prev, isSubscribed: false }));
        } catch (error) {
            console.error('Unsubscribe failed:', error);
        }
    }, []);

    return {
        ...state,
        requestPermission,
        subscribe,
        unsubscribe
    };
}
