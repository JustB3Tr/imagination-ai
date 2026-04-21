'use client';

import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import type {
  Attachment,
  Chat,
  DiffProposal,
  MediaArtifact,
  Message,
  ModelType,
  SummaryReport,
  TerminalRun,
  User,
  WorkspaceSnapshot,
} from './types';

export type AddMessageOptions = {
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
  sendChatToBackend: (userText: string, attachments?: Attachment[]) => Promise<void>;
  sendAgentChatToBackend: (userText: string, attachments?: Attachment[]) => Promise<void>;
  applyDiffProposals: (proposalIds: string[]) => Promise<void>;
  refreshWorkspaceTree: () => Promise<void>;
  setCurrentModel: (model: ModelType) => void;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => void;
  getCurrentChat: () => Chat | undefined;
  agentSessionId: string | null;
  workspaceSnapshot: WorkspaceSnapshot | null;
  terminalRuns: TerminalRun[];
  diffProposals: DiffProposal[];
  mediaArtifacts: MediaArtifact[];
  summaryReport: SummaryReport | null;
  isAgentRunning: boolean;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

const generateId = () => Math.random().toString(36).substring(2, 15);
const initialChats: Chat[] = [];

const OFFLINE_ASSISTANT_MESSAGE =
  'Imagination AI is currently offline. Please restart the backend.';

type AgentEvent = Record<string, unknown>;

export function ChatProvider({ children }: { children: ReactNode }) {
  const [chats, setChats] = useState<Chat[]>(initialChats);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [currentModel, setCurrentModel] = useState<ModelType>('imagination-1.3');
  const [user, setUser] = useState<User | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);

  const [agentSessionId, setAgentSessionId] = useState<string | null>(null);
  const [workspaceSnapshot, setWorkspaceSnapshot] = useState<WorkspaceSnapshot | null>(null);
  const [terminalRuns, setTerminalRuns] = useState<TerminalRun[]>([]);
  const [diffProposals, setDiffProposals] = useState<DiffProposal[]>([]);
  const [mediaArtifacts, setMediaArtifacts] = useState<MediaArtifact[]>([]);
  const [summaryReport, setSummaryReport] = useState<SummaryReport | null>(null);
  const [isAgentRunning, setIsAgentRunning] = useState(false);

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

  const selectChat = useCallback(
    (chatId: string) => {
      setCurrentChatId(chatId);
      const chat = chats.find(c => c.id === chatId);
      if (chat) {
        setCurrentModel(chat.model);
      }
    },
    [chats]
  );

  const deleteChat = useCallback(
    (chatId: string) => {
      setChats(prev => {
        const next = prev.filter(c => c.id !== chatId);
        if (currentChatId === chatId) {
          setCurrentChatId(next[0]?.id ?? null);
        }
        return next;
      });
    },
    [currentChatId]
  );

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

  const refreshWorkspaceTree = useCallback(async () => {
    if (!agentSessionId) return;
    try {
      const q = new URLSearchParams({ session_id: agentSessionId });
      const res = await fetch(`/api/agent/workspace?${q.toString()}`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      if (!res.ok) return;
      const data = (await res.json()) as WorkspaceSnapshot;
      setWorkspaceSnapshot(data);
    } catch {
      // best-effort refresh
    }
  }, [agentSessionId]);

  const handleAgentEvent = useCallback((event: AgentEvent) => {
    const type = String(event.type || '');
    if (type === 'session') {
      const sid = String(event.session_id || '');
      if (sid) setAgentSessionId(sid);
      return;
    }

    if (type === 'workspace_tree') {
      setWorkspaceSnapshot({
        root: String(event.root || ''),
        session_id: agentSessionId || String(event.session_id || ''),
        tree: (event.tree as WorkspaceSnapshot['tree']) ?? null,
        truncated: Boolean(event.truncated),
      });
      return;
    }

    if (type === 'tool_call' && String(event.name || '') === 'run_shell') {
      const id = String(event.id || generateId());
      const args = (event.args ?? {}) as Record<string, unknown>;
      const command = String(args.command || args.cmd || '');
      setTerminalRuns(prev => [...prev, { id, command, status: 'running' }]);
      return;
    }

    if (type === 'tool_result' && String(event.name || '') === 'run_shell') {
      const id = String(event.id || '');
      const data = ((event.data ?? {}) as Record<string, unknown>) || {};
      const status = String(data.status || 'fail') === 'success' ? 'success' : 'fail';
      setTerminalRuns(prev =>
        prev.map(run =>
          run.id === id
            ? {
                ...run,
                status,
                cwd: typeof data.cwd === 'string' ? data.cwd : run.cwd,
                stdout: typeof data.stdout === 'string' ? data.stdout : '',
                stderr: typeof data.stderr === 'string' ? data.stderr : '',
                exitCode: typeof data.exit_code === 'number' ? data.exit_code : undefined,
              }
            : run
        )
      );
      return;
    }

    if (type === 'diff_preview') {
      const proposalId = String(event.proposal_id || '');
      const path = String(event.path || '');
      const diff = String(event.diff || '');
      const applied = Boolean(event.applied);
      if (!proposalId) return;
      setDiffProposals(prev => {
        const has = prev.some(p => p.proposalId === proposalId);
        if (has) return prev;
        return [...prev, { proposalId, path, diff, applied }];
      });
      return;
    }

    if (type === 'media') {
      const kind = String(event.kind || 'screenshot');
      const sid = agentSessionId || String(event.session_id || '');
      const artifactId = String(event.artifact_id || '');
      if (!artifactId || !sid) return;

      if (kind === 'screenshot') {
        const base64 = String(event.base64 || '');
        const mimeType = String(event.mime_type || 'image/png');
        const src = base64
          ? `data:${mimeType};base64,${base64}`
          : `/api/agent/capture/${encodeURIComponent(sid)}/${encodeURIComponent(artifactId)}`;
        setMediaArtifacts(prev => [
          ...prev,
          { artifactId, sessionId: sid, kind: 'screenshot', mimeType, src },
        ]);

        const video = (event.video ?? null) as Record<string, unknown> | null;
        if (video && typeof video.artifact_id === 'string') {
          const videoArtifactId = video.artifact_id;
          const videoMimeType = typeof video.mime_type === 'string' ? video.mime_type : 'video/webm';
          const videoSrc = `/api/agent/capture/${encodeURIComponent(sid)}/${encodeURIComponent(videoArtifactId)}`;
          setMediaArtifacts(prev => [
            ...prev,
            {
              artifactId: videoArtifactId,
              sessionId: sid,
              kind: 'video',
              mimeType: videoMimeType,
              src: videoSrc,
            },
          ]);
        }
      }
      return;
    }

    if (type === 'summary') {
      const report = (event.report ?? null) as SummaryReport | null;
      if (report) {
        setSummaryReport(report);
      }
      return;
    }
  }, [agentSessionId]);

  const sendAgentChatToBackend = useCallback(
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
      setIsAgentRunning(true);
      setTerminalRuns([]);
      setDiffProposals([]);
      setMediaArtifacts([]);
      setSummaryReport(null);

      const body = {
        prompt: userText,
        currentModel,
        messages: history,
        session_id: agentSessionId ?? undefined,
        max_tool_iters: 28,
        confirm_apply: false,
        allow_network_tools: true,
      };

      let finalText = '';
      try {
        const res = await fetch('/api/chat/agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/x-ndjson' },
          body: JSON.stringify(body),
        });
        if (!res.ok || !res.body) {
          addMessage(OFFLINE_ASSISTANT_MESSAGE, 'assistant', undefined, { chatId: cid });
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            try {
              const event = JSON.parse(trimmed) as AgentEvent;
              handleAgentEvent(event);
              if (String(event.type || '') === 'final') {
                finalText = String(event.text || '').trim();
              }
            } catch {
              // ignore malformed lines
            }
          }
        }
      } catch {
        addMessage(OFFLINE_ASSISTANT_MESSAGE, 'assistant', undefined, { chatId: cid });
        return;
      } finally {
        setIsAgentRunning(false);
      }

      if (!finalText) {
        finalText = 'Agent loop completed. Review terminal output, diffs, and summary cards below.';
      }
      addMessage(finalText, 'assistant', undefined, { chatId: cid });
      await refreshWorkspaceTree();
    },
    [
      addMessage,
      agentSessionId,
      chats,
      createNewChat,
      currentChatId,
      currentModel,
      handleAgentEvent,
      refreshWorkspaceTree,
    ]
  );

  const sendChatToBackend = useCallback(
    async (userText: string, attachments?: Attachment[]) => {
      await sendAgentChatToBackend(userText, attachments);
    },
    [sendAgentChatToBackend]
  );

  const applyDiffProposals = useCallback(
    async (proposalIds: string[]) => {
      if (!agentSessionId || proposalIds.length === 0) return;
      const res = await fetch('/api/agent/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ session_id: agentSessionId, proposal_ids: proposalIds }),
      });
      if (!res.ok) return;
      const data = (await res.json()) as { applied?: Array<{ proposal_id: string }> };
      const appliedIds = new Set((data.applied || []).map(x => x.proposal_id));
      if (appliedIds.size === 0) return;
      setDiffProposals(prev =>
        prev.map(d => (appliedIds.has(d.proposalId) ? { ...d, applied: true } : d))
      );
      await refreshWorkspaceTree();
    },
    [agentSessionId, refreshWorkspaceTree]
  );

  const signIn = useCallback(async (email: string, _password: string) => {
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
        sendAgentChatToBackend,
        applyDiffProposals,
        refreshWorkspaceTree,
        setCurrentModel,
        signIn,
        signOut,
        getCurrentChat,
        agentSessionId,
        workspaceSnapshot,
        terminalRuns,
        diffProposals,
        mediaArtifacts,
        summaryReport,
        isAgentRunning,
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
