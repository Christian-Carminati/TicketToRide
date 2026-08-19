import { useEffect, useState, useCallback } from 'react';

type MessageHandler<T> = (data: T) => void;

class WebSocketSingleton {
  private static instances: Map<string, WebSocketSingleton> = new Map();
  private ws: WebSocket | null = null;
  private listeners: Set<MessageHandler<any>> = new Set();
  private statusListeners: Set<(connected: boolean) => void> = new Set();
  private reconnectTimer: number | null = null;
  public isConnected = false;
  private endpoint: string;

  private constructor(endpoint: string) {
    this.endpoint = endpoint;
    this.connect();
  }

  public static getInstance(endpoint: string): WebSocketSingleton {
    if (!WebSocketSingleton.instances.has(endpoint)) {
      WebSocketSingleton.instances.set(endpoint, new WebSocketSingleton(endpoint));
    }
    return WebSocketSingleton.instances.get(endpoint)!;
  }

  public subscribe(handler: MessageHandler<any>, onStatusChange: (connected: boolean) => void): () => void {
    this.listeners.add(handler);
    this.statusListeners.add(onStatusChange);
    onStatusChange(this.isConnected);

    if (!this.ws || this.ws.readyState === WebSocket.CLOSED) {
      this.connect();
    }

    return () => {
      this.listeners.delete(handler);
      this.statusListeners.delete(onStatusChange);
    };
  }

  public send(msg: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(typeof msg === 'string' ? msg : JSON.stringify(msg));
    }
  }

  private connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      return;
    }

    try {
      this.ws = new WebSocket(this.endpoint);

      this.ws.onopen = () => {
        this.isConnected = true;
        this.statusListeners.forEach((fn) => fn(true));
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.listeners.forEach((fn) => fn(data));
        } catch {
          // Ignore non-JSON
        }
      };

      this.ws.onclose = () => {
        this.isConnected = false;
        this.statusListeners.forEach((fn) => fn(false));
        if (this.listeners.size > 0 && !this.reconnectTimer) {
          this.reconnectTimer = window.setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
          }, 2000);
        }
      };

      this.ws.onerror = () => {
        if (this.ws) {
          this.ws.close();
        }
      };
    } catch {
      this.isConnected = false;
      this.statusListeners.forEach((fn) => fn(false));
    }
  }
}

export function useWebSocket<T = any>(endpoint = 'ws://localhost:8000/ws/telemetry') {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<T | null>(null);

  useEffect(() => {
    const singleton = WebSocketSingleton.getInstance(endpoint);
    const unsubscribe = singleton.subscribe(
      (data: T) => setLastMessage(data),
      (connected: boolean) => setIsConnected(connected)
    );
    return () => unsubscribe();
  }, [endpoint]);

  const sendMessage = useCallback((msg: any) => {
    WebSocketSingleton.getInstance(endpoint).send(msg);
  }, [endpoint]);

  return { isConnected, lastMessage, sendMessage };
}
