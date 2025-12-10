import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Minus, Send, Trash2, Loader2, Wrench, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useEngineeringChat, PartContext, Message } from '@/hooks/useEngineeringChat';
import { cn } from '@/lib/utils';

interface EngineeringChatWidgetProps {
  partContext?: PartContext;
}

// Clean any residual markdown formatting from AI responses
function cleanMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/`(.*?)`/g, '$1');
}

const quickQuestions = [
  "Heat treatment for 4140?",
  "Stock sizes for aluminum?",
  "Best HRC for gears?",
];

export function EngineeringChatWidget({ partContext }: EngineeringChatWidgetProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const { messages, isLoading, sendMessage, clearHistory } = useEngineeringChat();
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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

  // Floating button when closed
  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 h-14 w-14 rounded-full z-50 
          bg-gradient-to-br from-primary via-primary to-secondary
          shadow-[0_4px_20px_rgba(0,0,0,0.3),0_0_40px_rgba(180,39,45,0.2)]
          hover:shadow-[0_6px_30px_rgba(0,0,0,0.4),0_0_50px_rgba(180,39,45,0.3)]
          hover:scale-110 active:scale-95
          transition-all duration-300 ease-out
          flex items-center justify-center group"
      >
        <MessageCircle className="h-6 w-6 text-primary-foreground transition-transform group-hover:rotate-12" />
        <span className="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-green-500 
          animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
      </button>
    );
  }

  // Minimized state
  if (isMinimized) {
    return (
      <div className="fixed bottom-6 right-6 w-80 z-50 rounded-xl overflow-hidden
        shadow-[0_8px_32px_rgba(0,0,0,0.3)] border border-border/20">
        <div className="p-3 flex items-center justify-between 
          bg-gradient-to-r from-[hsl(220,15%,12%)] via-[hsl(215,12%,18%)] to-[hsl(210,10%,22%)]">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-primary to-primary/60 
              flex items-center justify-center shadow-inner">
              <Wrench className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="font-semibold text-sm text-white/90">Engineering Assistant</span>
          </div>
          <div className="flex gap-1">
            <button
              onClick={() => setIsMinimized(false)}
              className="h-7 w-7 rounded-md flex items-center justify-center
                text-white/60 hover:text-white hover:bg-white/10 transition-colors"
            >
              <Sparkles className="h-4 w-4" />
            </button>
            <button
              onClick={() => setIsOpen(false)}
              className="h-7 w-7 rounded-md flex items-center justify-center
                text-white/60 hover:text-white hover:bg-white/10 transition-colors"
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
    <div className="fixed bottom-6 right-6 w-[380px] h-[520px] z-50 flex flex-col
      rounded-2xl overflow-hidden
      bg-gradient-to-b from-background via-background to-muted/30
      shadow-[0_8px_40px_rgba(0,0,0,0.4),0_0_60px_rgba(180,39,45,0.1)]
      border border-border/30 backdrop-blur-sm">
      
      {/* Header */}
      <div className="p-4 flex items-center justify-between shrink-0
        bg-gradient-to-r from-[hsl(220,15%,12%)] via-[hsl(215,12%,18%)] to-[hsl(210,10%,22%)]
        border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-primary to-primary/60 
            flex items-center justify-center shadow-lg shadow-primary/20">
            <Wrench className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h3 className="font-semibold text-white/95 tracking-tight">Engineering Assistant</h3>
            {partContext?.name && (
              <p className="text-xs text-white/50 truncate max-w-[180px]">
                Viewing: {partContext.name}
              </p>
            )}
          </div>
        </div>
        <div className="flex gap-1">
          <button
            onClick={clearHistory}
            className="h-8 w-8 rounded-lg flex items-center justify-center
              text-white/50 hover:text-white hover:bg-white/10 transition-all"
            title="Clear history"
          >
            <Trash2 className="h-4 w-4" />
          </button>
          <button
            onClick={() => setIsMinimized(true)}
            className="h-8 w-8 rounded-lg flex items-center justify-center
              text-white/50 hover:text-white hover:bg-white/10 transition-all"
          >
            <Minus className="h-4 w-4" />
          </button>
          <button
            onClick={() => setIsOpen(false)}
            className="h-8 w-8 rounded-lg flex items-center justify-center
              text-white/50 hover:text-white hover:bg-white/10 transition-all"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full">
          <div ref={scrollRef} className="p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="text-center py-8 px-4">
                <div className="h-16 w-16 mx-auto mb-4 rounded-2xl 
                  bg-gradient-to-br from-primary/20 to-primary/5 
                  flex items-center justify-center border border-primary/20">
                  <Wrench className="h-8 w-8 text-primary/70" />
                </div>
                <p className="text-base font-medium text-foreground/80 mb-1">
                  Hey there! I'm your engineering buddy.
                </p>
                <p className="text-sm text-muted-foreground mb-6">
                  Ask me about heat treatment, stock sizes, HRC values, or calculations.
                </p>
                
                {/* Quick questions */}
                <div className="flex flex-wrap justify-center gap-2">
                  {quickQuestions.map((q, i) => (
                    <button
                      key={i}
                      onClick={() => handleSend(q)}
                      className="px-3 py-1.5 text-xs rounded-full
                        bg-gradient-to-r from-muted to-muted/50
                        hover:from-primary/20 hover:to-primary/10
                        border border-border/50 hover:border-primary/30
                        text-muted-foreground hover:text-foreground
                        transition-all duration-200"
                    >
                      {q}
                    </button>
                  ))}
                </div>
                
                {partContext?.name && (
                  <p className="text-xs mt-6 text-primary/70 font-medium">
                    Context: {partContext.name} ({partContext.material || 'Unknown material'})
                  </p>
                )}
              </div>
            ) : (
              messages.map((message, index) => (
                <MessageBubble key={index} message={message} />
              ))
            )}
            {isLoading && messages[messages.length - 1]?.role === 'user' && (
              <div className="flex items-center gap-2 text-muted-foreground text-sm pl-2">
                <div className="flex gap-1">
                  <span className="h-2 w-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="h-2 w-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="h-2 w-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-xs">Thinking...</span>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Input */}
      <div className="p-3 border-t border-border/30 shrink-0 bg-muted/20">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Input
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about materials, heat treatment..."
              disabled={isLoading}
              className="pr-4 bg-background/80 border-border/50 
                focus:border-primary/50 focus:ring-2 focus:ring-primary/20
                placeholder:text-muted-foreground/50"
            />
          </div>
          <Button
            onClick={() => handleSend()}
            disabled={!inputValue.trim() || isLoading}
            className="h-10 w-10 p-0 rounded-xl
              bg-gradient-to-br from-primary to-primary/80
              hover:from-primary hover:to-primary/90
              shadow-lg shadow-primary/20 hover:shadow-primary/30
              disabled:opacity-50 disabled:shadow-none
              transition-all duration-200"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
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
      'flex gap-2 animate-in fade-in-0 slide-in-from-bottom-2 duration-300',
      isUser ? 'justify-end' : 'justify-start'
    )}>
      {!isUser && (
        <div className="h-7 w-7 rounded-lg shrink-0 mt-1
          bg-gradient-to-br from-muted to-muted/50
          flex items-center justify-center border border-border/30">
          <Wrench className="h-3.5 w-3.5 text-muted-foreground" />
        </div>
      )}
      <div
        className={cn(
          'max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed',
          isUser
            ? 'bg-gradient-to-br from-primary to-primary/85 text-primary-foreground rounded-br-md shadow-lg shadow-primary/20'
            : 'bg-gradient-to-br from-muted/80 to-muted/40 text-foreground rounded-bl-md border border-border/30'
        )}
      >
        <div className="whitespace-pre-wrap break-words">
          {displayContent || (
            <span className="text-muted-foreground/70 italic">...</span>
          )}
        </div>
      </div>
    </div>
  );
}
