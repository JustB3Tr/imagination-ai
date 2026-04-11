'use client';

import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import type { Attachment, Chat, Message, ModelType, User } from './types';

export type AddMessageOptions = {
  /** When set, append to this chat instead of `currentChatId` (e.g. right after `createNewChat()`). */
  chatId?: string | null;
};

interface ChatContextType {
  chats: Chat[];
  currentChatId: string | null;
  currentModel: ModelType;
  user: User | null;
  isAuthenticated: boolean;
  showAuthModal: boolean;
  setShowAuthModal: (show: boolean) => void;
  createNewChat: () => string;
  selectChat: (chatId: string) => void;
  deleteChat: (chatId: string) => void;
  addMessage: (
    content: string,
    role: 'user' | 'assistant',
    attachments?: Attachment[],
    options?: AddMessageOptions
  ) => void;
  /**
   * POST same-origin `/api/chat` with `{ prompt, currentModel, messages }`.
   * The Route Handler proxies to Imagination using `NEXT_PUBLIC_API_URL` (see `app/api/chat/route.ts`).
   */
  sendChatToBackend: (userText: string, attachments?: Attachment[]) => Promise<void>;
  setCurrentModel: (model: ModelType) => void;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => void;
  getCurrentChat: () => Chat | undefined;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

const generateId = () => Math.random().toString(36).substring(2, 15);

const initialChats: Chat[] = [];

const OFFLINE_ASSISTANT_MESSAGE =
  'Imagination AI is currently offline. Please restart the Colab backend.';

export function ChatProvider({ children }: { children: ReactNode }) {
  const [chats, setChats] = useState<Chat[]>(initialChats);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [currentModel, setCurrentModel] = useState<ModelType>('imagination-1.3');
  const [user, setUser] = useState<User | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);

  const createNewChat = useCallback((): string => {
    const id = generateId();
    const newChat: Chat = {
      id,
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
      model: currentModel,
    };
    setChats(prev => [newChat, ...prev]);
    setCurrentChatId(id);
    return id;
  }, [currentModel]);

  const selectChat = useCallback((chatId: string) => {
    setCurrentChatId(chatId);
    const chat = chats.find(c => c.id === chatId);
    if (chat) {
      setCurrentModel(chat.model);
    }
  }, [chats]);

  const deleteChat = useCallback((chatId: string) => {
    setChats(prev => {
      const next = prev.filter(c => c.id !== chatId);
      if (currentChatId === chatId) {
        setCurrentChatId(next[0]?.id ?? null);
      }
      return next;
    });
  }, [currentChatId]);

  const addMessage = useCallback(
    (
      content: string,
      role: 'user' | 'assistant',
      attachments?: Attachment[],
      options?: AddMessageOptions
    ) => {
      const targetId = options?.chatId ?? currentChatId;
      if (!targetId) return;

      const newMessage: Message = {
        id: generateId(),
        role,
        content,
        attachments,
        createdAt: new Date(),
      };

      setChats(prev =>
        prev.map(chat => {
          if (chat.id === targetId) {
            const updatedMessages = [...chat.messages, newMessage];
            const title =
              chat.messages.length === 0 && role === 'user'
                ? content.slice(0, 30) + (content.length > 30 ? '...' : '')
                : chat.title;
            return { ...chat, messages: updatedMessages, title };
          }
          return chat;
        })
      );
    },
    [currentChatId]
  );

  const sendChatToBackend = useCallback(
    async (userText: string, attachments?: Attachment[]) => {
      let cid = currentChatId;
      if (!cid) {
        cid = createNewChat();
      }

      const priorMessages = (chats.find(c => c.id === cid)?.messages ?? []).map(m => ({
        role: m.role,
        content: m.content,
      }));

      let augmented = userText;
      if (attachments?.length) {
        const names = attachments.map(a => `${a.name} (${a.type})`).join(', ');
        augmented = `[Attached: ${names}]\n\n${userText}`;
      }

      const history = [...priorMessages, { role: 'user' as const, content: augmented }];

      addMessage(userText, 'user', attachments, { chatId: cid });

      const chatPayload = {
        prompt: userText,
        currentModel,
        messages: history.map(({ role, content }) => ({ role, content })),
      };

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify(chatPayload),
        });

        let data: { response?: string; error?: string; detail?: string } = {};
        const ct = res.headers.get('content-type') || '';
        if (ct.includes('application/json')) {
          try {
            data = (await res.json()) as typeof data;
          } catch {
            /* ignore */
          }
        }

        if (!res.ok) {
          const fallback =
            typeof data.response === 'string' && data.response.trim()
              ? data.response
              : OFFLINE_ASSISTANT_MESSAGE;
          addMessage(fallback, 'assistant', undefined, { chatId: cid });
          return;
        }

        const text =
          typeof data.response === 'string' ? data.response : '';
        addMessage(text.trim() ? text : '(Empty response)', 'assistant', undefined, {
          chatId: cid,
        });
      } catch {
        addMessage(OFFLINE_ASSISTANT_MESSAGE, 'assistant', undefined, { chatId: cid });
      }
    },
    [addMessage, chats, createNewChat, currentChatId, currentModel]
  );

  const signIn = useCallback(async (email: string, _password: string) => {
    // Simulate authentication
    await new Promise(resolve => setTimeout(resolve, 1000));
    setUser({
      id: generateId(),
      name: email.split('@')[0],
      email,
    });
    setShowAuthModal(false);
  }, []);

  const signOut = useCallback(() => {
    setUser(null);
  }, []);

  const getCurrentChat = useCallback(() => {
    return chats.find(c => c.id === currentChatId);
  }, [chats, currentChatId]);

  return (
    <ChatContext.Provider
      value={{
        chats,
        currentChatId,
        currentModel,
        user,
        isAuthenticated: !!user,
        showAuthModal,
        setShowAuthModal,
        createNewChat,
        selectChat,
        deleteChat,
        addMessage,
        sendChatToBackend,
        setCurrentModel,
        signIn,
        signOut,
        getCurrentChat,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChatContext() {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChatContext must be used within a ChatProvider');
  }
  return context;
}
