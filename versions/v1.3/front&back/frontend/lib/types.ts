export interface Attachment {
  id: string;
  name: string;
  type: 'image' | 'document' | 'code';
  mimeType: string;
  size: number;
  url: string; // Data URL or blob URL
  content?: string; // For text/code files
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  attachments?: Attachment[];
  createdAt: Date;
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  model: ModelType;
}

export type ModelType = 'imagination-1.3' | 'imagination-1.3-pro' | 'imagination-1.3-coder';

export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
}
