'use client';

import { Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ModelSelector } from './model-selector';
import { ChatMessages } from './chat-messages';
import { ChatInput } from './chat-input';

interface ChatAreaProps {
  onToggleSidebar: () => void;
  isSidebarOpen: boolean;
}

export function ChatArea({ onToggleSidebar, isSidebarOpen }: ChatAreaProps) {
  return (
    <div className="flex flex-col h-full flex-1 bg-background">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          {!isSidebarOpen && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggleSidebar}
              className="lg:hidden"
              aria-label="Open sidebar"
            >
              <Menu className="w-5 h-5" />
            </Button>
          )}
          <ModelSelector />
        </div>
        <div className="flex items-center gap-2">
          {/* Future: Add share, settings buttons here */}
        </div>
      </header>

      {/* Messages */}
      <ChatMessages />

      {/* Input */}
      <ChatInput />
    </div>
  );
}
