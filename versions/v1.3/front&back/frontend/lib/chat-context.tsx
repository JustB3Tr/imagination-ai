'use client';

import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import type { Attachment, Chat, Message, ModelType, User } from './types';

interface ChatContextType {
  chats: Chat[];
  currentChatId: string | null;
  currentModel: ModelType;
  user: User | null;
  isAuthenticated: boolean;
  showAuthModal: boolean;
  setShowAuthModal: (show: boolean) => void;
  createNewChat: () => void;
  selectChat: (chatId: string) => void;
  deleteChat: (chatId: string) => void;
  addMessage: (content: string, role: 'user' | 'assistant', attachments?: Attachment[]) => void;
  setCurrentModel: (model: ModelType) => void;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => void;
  getCurrentChat: () => Chat | undefined;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

const generateId = () => Math.random().toString(36).substring(2, 15);

// Demo message with KaTeX math rendering
const demoMessages: Message[] = [
  {
    id: generateId(),
    role: 'user',
    content: 'Can you explain how to solve a simple fraction like dividing 3 by -2?',
    createdAt: new Date(Date.now() - 60000),
  },
  {
    id: generateId(),
    role: 'assistant',
    content: `Of course! When you divide 3 by -2, you get a negative fraction.

The result is: $\\frac{3}{-2}$

This can also be written as $-\\frac{3}{2}$ or approximately $-1.5$.

**Key points to remember:**
- When dividing a positive by a negative, the result is negative
- The fraction $\\frac{3}{-2}$ equals $\\frac{-3}{2}$ equals $-\\frac{3}{2}$
- In decimal form: $-1.5$

Here's another example with the quadratic formula: $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$`,
    createdAt: new Date(),
  },
];

const initialChats: Chat[] = [
  {
    id: generateId(),
    title: 'Math Fractions Help',
    messages: demoMessages,
    createdAt: new Date(),
    model: 'imagination-1.3',
  },
  {
    id: generateId(),
    title: 'Code Review Assistant',
    messages: [],
    createdAt: new Date(Date.now() - 86400000),
    model: 'imagination-1.3-coder',
  },
  {
    id: generateId(),
    title: 'Creative Writing Ideas',
    messages: [],
    createdAt: new Date(Date.now() - 172800000),
    model: 'imagination-1.3-pro',
  },
];

export function ChatProvider({ children }: { children: ReactNode }) {
  const [chats, setChats] = useState<Chat[]>(initialChats);
  const [currentChatId, setCurrentChatId] = useState<string | null>(initialChats[0].id);
  const [currentModel, setCurrentModel] = useState<ModelType>('imagination-1.3');
  const [user, setUser] = useState<User | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);

  const createNewChat = useCallback(() => {
    const newChat: Chat = {
      id: generateId(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
      model: currentModel,
    };
    setChats(prev => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
  }, [currentModel]);

  const selectChat = useCallback((chatId: string) => {
    setCurrentChatId(chatId);
    const chat = chats.find(c => c.id === chatId);
    if (chat) {
      setCurrentModel(chat.model);
    }
  }, [chats]);

  const deleteChat = useCallback((chatId: string) => {
    setChats(prev => prev.filter(c => c.id !== chatId));
    if (currentChatId === chatId) {
      setCurrentChatId(chats.length > 1 ? chats.find(c => c.id !== chatId)?.id || null : null);
    }
  }, [currentChatId, chats]);

  const addMessage = useCallback((content: string, role: 'user' | 'assistant', attachments?: Attachment[]) => {
    if (!currentChatId) return;

    const newMessage: Message = {
      id: generateId(),
      role,
      content,
      attachments,
      createdAt: new Date(),
    };

    setChats(prev => prev.map(chat => {
      if (chat.id === currentChatId) {
        const updatedMessages = [...chat.messages, newMessage];
        const title = chat.messages.length === 0 && role === 'user' 
          ? content.slice(0, 30) + (content.length > 30 ? '...' : '')
          : chat.title;
        return { ...chat, messages: updatedMessages, title };
      }
      return chat;
    }));
  }, [currentChatId]);

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
