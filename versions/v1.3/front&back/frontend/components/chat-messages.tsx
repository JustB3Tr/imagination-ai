'use client';

import { useRef, useEffect } from 'react';
import { Bot, User, Copy, Check, FileText, FileCode, Image as ImageIcon } from 'lucide-react';
import type { Attachment } from '@/lib/types';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { useChatContext } from '@/lib/chat-context';
import { MathRenderer } from './math-renderer';
import { cn } from '@/lib/utils';
import { useState } from 'react';

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function AttachmentPreview({ attachment }: { attachment: Attachment }) {
  const getIcon = () => {
    switch (attachment.type) {
      case 'image': return <ImageIcon className="w-4 h-4" />;
      case 'code': return <FileCode className="w-4 h-4" />;
      default: return <FileText className="w-4 h-4" />;
    }
  };

  if (attachment.type === 'image') {
    return (
      <div className="relative rounded-lg overflow-hidden max-w-xs">
        <img
          src={attachment.url}
          alt={attachment.name}
          className="w-full h-auto max-h-64 object-cover"
        />
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-2">
          <p className="text-white text-xs truncate">{attachment.name}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 bg-muted/50 rounded-lg p-3 max-w-xs">
      <div className="flex items-center justify-center w-10 h-10 bg-secondary rounded-lg shrink-0">
        {getIcon()}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-foreground truncate">{attachment.name}</p>
        <p className="text-xs text-muted-foreground">{formatFileSize(attachment.size)}</p>
      </div>
    </div>
  );
}

function CopyButton({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={handleCopy}
      className="opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8 p-0"
    >
      {copied ? (
        <Check className="w-4 h-4 text-accent" />
      ) : (
        <Copy className="w-4 h-4 text-muted-foreground" />
      )}
    </Button>
  );
}

export function ChatMessages() {
  const { getCurrentChat } = useChatContext();
  const chat = getCurrentChat();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chat?.messages]);

  if (!chat || (chat.messages?.length ?? 0) === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center max-w-md px-4">
          <div className="flex items-center justify-center w-16 h-16 mx-auto mb-6 rounded-2xl bg-accent/10">
            <Bot className="w-8 h-8 text-accent" />
          </div>
          <h2 className="text-2xl font-semibold text-foreground mb-2">
            How can I help you today?
          </h2>
          <p className="text-muted-foreground">
            Start a conversation with Imagination AI. Ask questions, get creative, or explore ideas together.
          </p>
        </div>
      </div>
    );
  }

  return (
    <ScrollArea ref={scrollRef} className="flex-1 px-4">
      <div className="max-w-3xl mx-auto py-8 space-y-6">
        {(chat.messages ?? []).map((message) => (
          <div
            key={message.id}
            className={cn(
              'group flex gap-4',
              message.role === 'user' ? 'flex-row-reverse' : ''
            )}
          >
            {/* Avatar */}
            <div
              className={cn(
                'flex items-center justify-center w-8 h-8 rounded-full shrink-0',
                message.role === 'user'
                  ? 'bg-accent text-accent-foreground'
                  : 'bg-secondary text-secondary-foreground'
              )}
            >
              {message.role === 'user' ? (
                <User className="w-4 h-4" />
              ) : (
                <Bot className="w-4 h-4" />
              )}
            </div>

            {/* Message Bubble */}
            <div
              className={cn(
                'flex-1 max-w-[80%]',
                message.role === 'user' ? 'flex justify-end' : ''
              )}
            >
              <div
                className={cn(
                  'relative px-4 py-3 rounded-2xl',
                  message.role === 'user'
                    ? 'bg-accent text-accent-foreground rounded-tr-md'
                    : 'bg-card border border-border rounded-tl-md'
                )}
              >
                {/* Attachments */}
                {message.attachments && message.attachments.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-3">
                    {message.attachments?.map((attachment) => (
                      <AttachmentPreview key={attachment.id} attachment={attachment} />
                    ))}
                  </div>
                )}
                
                {message.role === 'assistant' ? (
                  <MathRenderer content={message.content} />
                ) : (
                  message.content && <p className="whitespace-pre-wrap">{message.content}</p>
                )}
              </div>

              {/* Actions */}
              {message.role === 'assistant' && (
                <div className="flex items-center gap-1 mt-1 px-2">
                  <CopyButton content={message.content} />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}
