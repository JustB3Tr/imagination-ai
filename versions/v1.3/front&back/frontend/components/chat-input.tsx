'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Mic, X, FileText, FileCode, Image as ImageIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useChatContext } from '@/lib/chat-context';
import type { Attachment } from '@/lib/types';

// Accepted file types
const ACCEPTED_TYPES = {
  image: ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml'],
  document: ['text/plain', 'text/markdown', 'application/pdf', 'text/csv'],
  code: [
    'text/javascript', 'application/javascript',
    'text/typescript', 'application/typescript',
    'text/html', 'text/css', 'application/json',
    'text/x-python', 'application/x-python',
    'text/x-java', 'text/x-c', 'text/x-c++',
  ],
};

const CODE_EXTENSIONS = ['.js', '.jsx', '.ts', '.tsx', '.py', '.java', '.c', '.cpp', '.h', '.css', '.html', '.json', '.md', '.xml', '.yaml', '.yml', '.sh', '.sql', '.go', '.rs', '.rb', '.php'];
const DOC_EXTENSIONS = ['.txt', '.md', '.csv', '.pdf'];
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'];

function getFileType(file: File): 'image' | 'document' | 'code' {
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  
  if (IMAGE_EXTENSIONS.includes(ext) || ACCEPTED_TYPES.image.includes(file.type)) {
    return 'image';
  }
  if (CODE_EXTENSIONS.includes(ext)) {
    return 'code';
  }
  if (DOC_EXTENSIONS.includes(ext) || ACCEPTED_TYPES.document.includes(file.type)) {
    return 'document';
  }
  // Default to document for unknown text files
  if (file.type.startsWith('text/')) {
    return 'document';
  }
  return 'document';
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Smart math response generator
function generateMathResponse(userMessage: string, attachments?: Attachment[]): string {
  const lowerMessage = userMessage.toLowerCase();
  
  // Check if there are attachments
  if (attachments && attachments.length > 0) {
    const attachmentDescriptions = attachments.map(a => {
      if (a.type === 'image') return `an image "${a.name}"`;
      if (a.type === 'code') return `a code file "${a.name}"`;
      return `a document "${a.name}"`;
    }).join(', ');
    
    const hasCode = attachments.some(a => a.type === 'code');
    const hasImage = attachments.some(a => a.type === 'image');
    
    if (hasCode) {
      const codeFile = attachments.find(a => a.type === 'code');
      return `I can see you've shared ${attachmentDescriptions}.\n\n**File Analysis:**\n- **Name:** ${codeFile?.name}\n- **Size:** ${formatFileSize(codeFile?.size || 0)}\n\nI'd be happy to help you with this code! Here are some things I can do:\n\n1. **Review** the code for potential issues\n2. **Explain** what the code does\n3. **Optimize** or refactor it\n4. **Debug** any problems\n\nWhat would you like me to help with?`;
    }
    
    if (hasImage) {
      return `I can see you've shared ${attachmentDescriptions}.\n\nI can analyze images for:\n- **Content description** - What's in the image\n- **Text extraction** (OCR)\n- **Code from screenshots**\n- **Design feedback**\n\nHow can I help you with this image?`;
    }
    
    return `I've received ${attachmentDescriptions}.\n\nI can help you:\n- **Summarize** the content\n- **Extract** key information\n- **Answer questions** about it\n\nWhat would you like to know?`;
  }
  
  // Square root questions
  if (lowerMessage.includes('square root') || lowerMessage.includes('sqrt')) {
    const numberMatch = userMessage.match(/\d+/);
    if (numberMatch) {
      const num = parseInt(numberMatch[0]);
      const sqrt = Math.sqrt(num);
      const isWhole = Number.isInteger(sqrt);
      return `The square root of ${num} is:\n\n$$\\sqrt{${num}} = ${isWhole ? sqrt : sqrt.toFixed(4)}$$\n\n${isWhole ? `This is a perfect square because $${sqrt} \\times ${sqrt} = ${num}$` : `This is not a perfect square. The result is approximately $${sqrt.toFixed(4)}$`}`;
    }
  }
  
  // Fraction questions
  if (lowerMessage.includes('fraction') || lowerMessage.includes('divide') || lowerMessage.includes('divided by')) {
    const numbers = userMessage.match(/-?\d+/g);
    if (numbers && numbers.length >= 2) {
      const a = parseInt(numbers[0]);
      const b = parseInt(numbers[1]);
      if (b !== 0) {
        return `When you divide ${a} by ${b}, you get:\n\n$$\\frac{${a}}{${b}} = ${(a/b).toFixed(4)}$$\n\nThis can be written as the fraction $\\frac{${a}}{${b}}$ or approximately $${(a/b).toFixed(2)}$ in decimal form.`;
      }
    }
  }
  
  // Power/exponent questions
  if (lowerMessage.includes('power') || lowerMessage.includes('squared') || lowerMessage.includes('cubed') || lowerMessage.includes('^')) {
    const numbers = userMessage.match(/\d+/g);
    if (numbers && numbers.length >= 1) {
      const base = parseInt(numbers[0]);
      const exp = lowerMessage.includes('cubed') ? 3 : lowerMessage.includes('squared') ? 2 : (numbers[1] ? parseInt(numbers[1]) : 2);
      return `The result of ${base} raised to the power of ${exp} is:\n\n$$${base}^{${exp}} = ${Math.pow(base, exp)}$$\n\nThis means multiplying ${base} by itself ${exp} times.`;
    }
  }
  
  // General math keywords
  if (lowerMessage.includes('quadratic') || lowerMessage.includes('equation')) {
    return `The **quadratic formula** is used to solve equations of the form $ax^2 + bx + c = 0$:\n\n$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$\n\nWhere:\n- $a$, $b$, and $c$ are coefficients\n- The $\\pm$ indicates two solutions\n- $b^2 - 4ac$ is called the discriminant`;
  }
  
  if (lowerMessage.includes('pythagorean') || lowerMessage.includes('triangle')) {
    return `The **Pythagorean theorem** states that for a right triangle:\n\n$$a^2 + b^2 = c^2$$\n\nWhere $c$ is the hypotenuse (longest side) and $a$ and $b$ are the other two sides.\n\nTo find the hypotenuse: $c = \\sqrt{a^2 + b^2}$`;
  }
  
  if (lowerMessage.includes('derivative') || lowerMessage.includes('calculus')) {
    return `Here are some common **derivative** rules:\n\n**Power Rule:**\n$$\\frac{d}{dx}x^n = nx^{n-1}$$\n\n**Product Rule:**\n$$\\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$$\n\n**Chain Rule:**\n$$\\frac{d}{dx}f(g(x)) = f'(g(x)) \\cdot g'(x)$$`;
  }
  
  if (lowerMessage.includes('integral') || lowerMessage.includes('integrate')) {
    return `Here's a famous integral - the **Gaussian integral**:\n\n$$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$\n\nAnd the general power rule for integration:\n\n$$\\int x^n dx = \\frac{x^{n+1}}{n+1} + C$$\n\nwhere $C$ is the constant of integration.`;
  }
  
  // Simple arithmetic
  const mathPattern = /(\d+)\s*([+\-*/×÷])\s*(\d+)/;
  const match = userMessage.match(mathPattern);
  if (match) {
    const a = parseFloat(match[1]);
    const op = match[2];
    const b = parseFloat(match[3]);
    let result: number;
    let opSymbol: string;
    
    switch (op) {
      case '+': result = a + b; opSymbol = '+'; break;
      case '-': result = a - b; opSymbol = '-'; break;
      case '*': case '×': result = a * b; opSymbol = '\\times'; break;
      case '/': case '÷': result = a / b; opSymbol = '\\div'; break;
      default: result = 0; opSymbol = op;
    }
    
    return `The answer is:\n\n$$${a} ${opSymbol} ${b} = ${result}$$`;
  }
  
  // Default response
  return `I'd be happy to help! Here's some mathematical context:\n\nEuler's identity is often called the most beautiful equation in mathematics:\n\n$$e^{i\\pi} + 1 = 0$$\n\nThis remarkable equation connects five fundamental constants:\n- $e$ (Euler's number)\n- $i$ (imaginary unit)\n- $\\pi$ (pi)\n- $1$ (unity)\n- $0$ (zero)\n\nFeel free to ask me any math questions, and I'll show you the formulas!`;
}

export function ChatInput() {
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { addMessage, currentChatId, createNewChat } = useChatContext();

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const newAttachments: Attachment[] = [];

    for (const file of Array.from(files)) {
      // Max 10MB per file
      if (file.size > 10 * 1024 * 1024) {
        alert(`File "${file.name}" is too large. Maximum size is 10MB.`);
        continue;
      }

      const fileType = getFileType(file);
      
      // Read file content
      const url = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        if (fileType === 'image') {
          reader.readAsDataURL(file);
        } else {
          reader.readAsDataURL(file);
        }
      });

      // For text files, also read the content
      let content: string | undefined;
      if (fileType === 'code' || fileType === 'document') {
        content = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.readAsText(file);
        });
      }

      newAttachments.push({
        id: Math.random().toString(36).substring(2, 15),
        name: file.name,
        type: fileType,
        mimeType: file.type,
        size: file.size,
        url,
        content,
      });
    }

    setAttachments(prev => [...prev, ...newAttachments]);
    
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeAttachment = (id: string) => {
    setAttachments(prev => prev.filter(a => a.id !== id));
  };

  const handleSubmit = async () => {
    if ((!input.trim() && attachments.length === 0) || isLoading) return;

    if (!currentChatId) {
      createNewChat();
    }

    const userMessage = input.trim();
    const currentAttachments = [...attachments];
    setInput('');
    setAttachments([]);
    setIsLoading(true);

    // Add user message with attachments
    addMessage(userMessage || 'Attached files', 'user', currentAttachments.length > 0 ? currentAttachments : undefined);

    // Simulate AI response
    setTimeout(() => {
      const response = generateMathResponse(userMessage, currentAttachments);
      addMessage(response, 'assistant');
      setIsLoading(false);
    }, 1500);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const getAttachmentIcon = (type: Attachment['type']) => {
    switch (type) {
      case 'image': return <ImageIcon className="w-4 h-4" />;
      case 'code': return <FileCode className="w-4 h-4" />;
      default: return <FileText className="w-4 h-4" />;
    }
  };

  return (
    <div className="border-t border-border bg-background p-4">
      <div className="max-w-3xl mx-auto">
        {/* Attachment previews */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {attachments.map((attachment) => (
              <div
                key={attachment.id}
                className="relative group flex items-center gap-2 bg-secondary rounded-lg p-2 pr-8"
              >
                {attachment.type === 'image' ? (
                  <img
                    src={attachment.url}
                    alt={attachment.name}
                    className="w-10 h-10 object-cover rounded"
                  />
                ) : (
                  <div className="w-10 h-10 flex items-center justify-center bg-muted rounded">
                    {getAttachmentIcon(attachment.type)}
                  </div>
                )}
                <div className="flex flex-col min-w-0">
                  <span className="text-sm text-foreground truncate max-w-[150px]">
                    {attachment.name}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {formatFileSize(attachment.size)}
                  </span>
                </div>
                <button
                  onClick={() => removeAttachment(attachment.id)}
                  className="absolute top-1 right-1 p-1 rounded-full bg-muted hover:bg-destructive hover:text-destructive-foreground transition-colors"
                  aria-label={`Remove ${attachment.name}`}
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="relative flex items-end gap-2 bg-card rounded-2xl border border-border p-2">
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.gif,.webp,.svg,.txt,.md,.pdf,.csv,.js,.jsx,.ts,.tsx,.py,.java,.c,.cpp,.h,.css,.html,.json,.xml,.yaml,.yml,.sh,.sql,.go,.rs,.rb,.php"
            onChange={handleFileSelect}
            className="hidden"
          />
          
          <Button
            variant="ghost"
            size="icon"
            className="shrink-0 text-muted-foreground hover:text-foreground"
            aria-label="Attach file"
            onClick={() => fileInputRef.current?.click()}
          >
            <Paperclip className="w-5 h-5" />
          </Button>

          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message Imagination..."
            rows={1}
            className="flex-1 resize-none bg-transparent border-0 focus:outline-none focus:ring-0 text-foreground placeholder:text-muted-foreground py-2 px-2 max-h-[200px]"
          />

          <div className="flex items-center gap-1 shrink-0">
            <Button
              variant="ghost"
              size="icon"
              className="text-muted-foreground hover:text-foreground"
              aria-label="Voice input"
            >
              <Mic className="w-5 h-5" />
            </Button>

            <Button
              onClick={handleSubmit}
              disabled={(!input.trim() && attachments.length === 0) || isLoading}
              size="icon"
              className="bg-accent text-accent-foreground hover:bg-accent/90 disabled:opacity-50"
              aria-label="Send message"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>

        <p className="text-center text-xs text-muted-foreground mt-3">
          Imagination AI can make mistakes. Consider checking important information.
        </p>
      </div>
    </div>
  );
}
