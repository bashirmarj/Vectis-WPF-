import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, ChevronDown, Send, RotateCcw, Loader2, Bot, User } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useEngineeringChat, PartContext, Message } from '@/hooks/useEngineeringChat';
import { cn } from '@/lib/utils';

interface EngineeringChatWidgetProps {
  partContext?: PartContext;
}

function cleanMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/`(.*?)`/g, '$1');
}

const quickPrompts = [
  "What heat treatment for 4140 steel?",
  "Standard aluminum bar sizes",
  "Recommended HRC for gears",
];

export function EngineeringChatWidget({ partContext }: EngineeringChatWidgetProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const { messages, isLoading, sendMessage, clearHistory } = useEngineeringChat();
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (isOpen && !isMinimized && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen, isMinimized]);

  const handleSend = (text?: string) => {
    const messageText = text || inputValue.trim();
    if (messageText && !isLoading) {
      sendMessage(messageText, partContext);
      setInputValue('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Floating launcher button
  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 group"
        aria-label="Open engineering assistant"
      >
        <div className="relative">
          {/* Glow effect */}
          <div className="absolute inset-0 rounded-full bg-primary/30 blur-lg group-hover:bg-primary/50 transition-all duration-300" />
          
          {/* Main button */}
          <div className="relative h-14 w-14 rounded-full 
            bg-gradient-to-br from-primary to-[hsl(355,70%,35%)]
            shadow-xl shadow-primary/25
            flex items-center justify-center
            transition-all duration-300
            group-hover:scale-110 group-hover:shadow-2xl group-hover:shadow-primary/40
            group-active:scale-95">
            <MessageCircle className="h-6 w-6 text-white transition-transform duration-300 group-hover:scale-110" />
          </div>
          
          {/* Online indicator */}
          <span className="absolute top-0 right-0 h-4 w-4 rounded-full 
            bg-emerald-500 border-2 border-background
            shadow-lg shadow-emerald-500/50" />
        </div>
        
        {/* Tooltip */}
        <div className="absolute bottom-full right-0 mb-2 px-3 py-1.5 
          bg-foreground text-background text-sm font-medium rounded-lg
          opacity-0 group-hover:opacity-100 transition-opacity duration-200
          whitespace-nowrap pointer-events-none">
          Engineering Assistant
        </div>
      </button>
    );
  }

  // Minimized state - just header bar
  if (isMinimized) {
    return (
      <div className="fixed bottom-6 right-6 z-50 w-80
        rounded-2xl overflow-hidden
        bg-gradient-to-r from-[hsl(220,20%,14%)] to-[hsl(220,15%,20%)]
        shadow-2xl shadow-black/40
        border border-white/10
        animate-in slide-in-from-bottom-4 duration-300">
        <div className="p-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl 
              bg-gradient-to-br from-primary to-primary/70
              flex items-center justify-center shadow-lg shadow-primary/20">
              <Bot className="h-5 w-5 text-white" />
            </div>
            <div>
              <span className="font-semibold text-white/95 text-sm">Engineering Assistant</span>
              <div className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-xs text-white/50">Online</span>
              </div>
            </div>
          </div>
          <div className="flex gap-1">
            <button
              onClick={() => setIsMinimized(false)}
              className="h-8 w-8 rounded-lg flex items-center justify-center
                text-white/60 hover:text-white hover:bg-white/10 transition-all"
            >
              <ChevronDown className="h-4 w-4 rotate-180" />
            </button>
            <button
              onClick={() => setIsOpen(false)}
              className="h-8 w-8 rounded-lg flex items-center justify-center
                text-white/60 hover:text-white hover:bg-white/10 transition-all"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Full chat panel
  return (
    <div className="fixed bottom-6 right-6 z-50 w-[400px] h-[600px]
      flex flex-col rounded-2xl overflow-hidden
      bg-background
      shadow-2xl shadow-black/30
      border border-border/50
      animate-in slide-in-from-bottom-4 fade-in duration-300">
      
      {/* Header */}
      <div className="shrink-0 bg-gradient-to-r from-[hsl(220,20%,14%)] to-[hsl(220,15%,22%)]">
        <div className="p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="h-11 w-11 rounded-xl 
                bg-gradient-to-br from-primary via-primary to-[hsl(355,65%,40%)]
                flex items-center justify-center shadow-lg shadow-primary/30">
                <Bot className="h-6 w-6 text-white" />
              </div>
              <span className="absolute -bottom-0.5 -right-0.5 h-3.5 w-3.5 rounded-full 
                bg-emerald-400 border-2 border-[hsl(220,20%,14%)]" />
            </div>
            <div>
              <h3 className="font-semibold text-white tracking-tight">Engineering Assistant</h3>
              <p className="text-xs text-white/50">
                {partContext?.name ? `Viewing: ${partContext.name}` : 'Ask me anything about manufacturing'}
              </p>
            </div>
          </div>
          <div className="flex gap-1">
            <button
              onClick={clearHistory}
              className="h-9 w-9 rounded-lg flex items-center justify-center
                text-white/40 hover:text-white hover:bg-white/10 transition-all"
              title="Clear conversation"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
            <button
              onClick={() => setIsMinimized(true)}
              className="h-9 w-9 rounded-lg flex items-center justify-center
                text-white/40 hover:text-white hover:bg-white/10 transition-all"
            >
              <ChevronDown className="h-4 w-4" />
            </button>
            <button
              onClick={() => setIsOpen(false)}
              className="h-9 w-9 rounded-lg flex items-center justify-center
                text-white/40 hover:text-white hover:bg-white/10 transition-all"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-hidden bg-gradient-to-b from-muted/30 to-background">
        <ScrollArea className="h-full">
          <div ref={scrollRef} className="p-4 space-y-4">
            {messages.length === 0 ? (
              <EmptyState 
                onPromptClick={handleSend} 
                partContext={partContext}
              />
            ) : (
              <>
                {messages.map((message, index) => (
                  <MessageBubble key={index} message={message} />
                ))}
                {isLoading && messages[messages.length - 1]?.role === 'user' && (
                  <TypingIndicator />
                )}
              </>
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Input area */}
      <div className="shrink-0 p-4 border-t border-border/50 bg-background">
        <div className="relative flex items-end gap-2">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your question..."
              disabled={isLoading}
              rows={1}
              className="w-full resize-none rounded-xl 
                bg-muted/50 border border-border/50
                px-4 py-3 pr-12
                text-sm text-foreground placeholder:text-muted-foreground/60
                focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50
                disabled:opacity-50
                transition-all duration-200"
              style={{ maxHeight: '120px' }}
            />
          </div>
          <button
            onClick={() => handleSend()}
            disabled={!inputValue.trim() || isLoading}
            className="h-11 w-11 rounded-xl shrink-0
              bg-gradient-to-br from-primary to-[hsl(355,65%,38%)]
              flex items-center justify-center
              shadow-lg shadow-primary/20
              disabled:opacity-40 disabled:shadow-none disabled:cursor-not-allowed
              hover:shadow-xl hover:shadow-primary/30 hover:scale-105
              active:scale-95
              transition-all duration-200"
          >
            {isLoading ? (
              <Loader2 className="h-5 w-5 text-white animate-spin" />
            ) : (
              <Send className="h-5 w-5 text-white" />
            )}
          </button>
        </div>
        <p className="text-[10px] text-muted-foreground/50 text-center mt-2">
          Powered by AI • Heat treatment, stock sizes, calculations
        </p>
      </div>
    </div>
  );
}

function EmptyState({ 
  onPromptClick, 
  partContext 
}: { 
  onPromptClick: (text: string) => void;
  partContext?: PartContext;
}) {
  return (
    <div className="py-8 px-2">
      {/* Welcome message bubble */}
      <div className="flex gap-3 mb-6">
        <div className="h-9 w-9 rounded-xl shrink-0
          bg-gradient-to-br from-primary/20 to-primary/5
          flex items-center justify-center border border-primary/20">
          <Bot className="h-5 w-5 text-primary" />
        </div>
        <div className="flex-1 bg-muted/60 rounded-2xl rounded-tl-md p-4 border border-border/30">
          <p className="text-sm text-foreground leading-relaxed">
            Hey there! I'm your engineering assistant. I can help with heat treatment recommendations, 
            material stock sizes, HRC values, and engineering calculations. What can I help you with?
          </p>
          {partContext?.name && (
            <p className="text-xs text-primary mt-2 font-medium">
              I see you're working with {partContext.name}
              {partContext.material && ` in ${partContext.material}`}. Feel free to ask about it!
            </p>
          )}
        </div>
      </div>

      {/* Quick prompts */}
      <div className="space-y-2">
        <p className="text-xs text-muted-foreground font-medium px-1 mb-3">Try asking:</p>
        {quickPrompts.map((prompt, i) => (
          <button
            key={i}
            onClick={() => onPromptClick(prompt)}
            className="w-full text-left px-4 py-3 rounded-xl
              bg-background hover:bg-muted/80
              border border-border/50 hover:border-primary/30
              text-sm text-foreground/80 hover:text-foreground
              transition-all duration-200
              hover:shadow-sm hover:translate-x-1"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="h-9 w-9 rounded-xl shrink-0
        bg-gradient-to-br from-primary/20 to-primary/5
        flex items-center justify-center border border-primary/20">
        <Bot className="h-5 w-5 text-primary" />
      </div>
      <div className="bg-muted/60 rounded-2xl rounded-tl-md px-4 py-3 border border-border/30">
        <div className="flex gap-1.5 items-center h-5">
          <span className="h-2 w-2 rounded-full bg-foreground/40 animate-bounce" 
            style={{ animationDelay: '0ms', animationDuration: '600ms' }} />
          <span className="h-2 w-2 rounded-full bg-foreground/40 animate-bounce" 
            style={{ animationDelay: '150ms', animationDuration: '600ms' }} />
          <span className="h-2 w-2 rounded-full bg-foreground/40 animate-bounce" 
            style={{ animationDelay: '300ms', animationDuration: '600ms' }} />
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';
  const displayContent = isUser ? message.content : cleanMarkdown(message.content);
  
  return (
    <div className={cn(
      'flex gap-3 animate-in fade-in-0 slide-in-from-bottom-2 duration-300',
      isUser ? 'flex-row-reverse' : 'flex-row'
    )}>
      {/* Avatar */}
      <div className={cn(
        'h-9 w-9 rounded-xl shrink-0 flex items-center justify-center',
        isUser 
          ? 'bg-gradient-to-br from-primary to-primary/70 shadow-lg shadow-primary/20'
          : 'bg-gradient-to-br from-primary/20 to-primary/5 border border-primary/20'
      )}>
        {isUser ? (
          <User className="h-5 w-5 text-white" />
        ) : (
          <Bot className="h-5 w-5 text-primary" />
        )}
      </div>

      {/* Message bubble */}
      <div
        className={cn(
          'max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed',
          isUser
            ? 'bg-gradient-to-br from-primary to-[hsl(355,65%,38%)] text-white rounded-tr-md shadow-lg shadow-primary/20'
            : 'bg-muted/60 text-foreground rounded-tl-md border border-border/30'
        )}
      >
        <div className="whitespace-pre-wrap break-words">
          {displayContent || (
            <span className="opacity-50 italic">...</span>
          )}
        </div>
      </div>
    </div>
  );
}
