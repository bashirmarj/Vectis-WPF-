import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, ChevronDown, Send, RotateCcw, Loader2, User, Sparkles, Wrench, Ruler, Flame } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useEngineeringChat, PartContext, Message } from '@/hooks/useEngineeringChat';
import { cn } from '@/lib/utils';
import logo from '@/assets/logo.png';

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
  { icon: Flame, title: "Heat Treatment", prompt: "What heat treatment for 4140 steel?" },
  { icon: Ruler, title: "Stock Sizes", prompt: "Standard aluminum bar sizes" },
  { icon: Wrench, title: "HRC Values", prompt: "Recommended HRC for gears" },
];

export function EngineeringChatWidget({ partContext }: EngineeringChatWidgetProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [showSendPulse, setShowSendPulse] = useState(false);
  const { messages, isLoading, sendMessage, clearHistory } = useEngineeringChat();
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [messages]);

  useEffect(() => {
    if (isOpen && !isMinimized && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen, isMinimized]);

  const handleSend = (text?: string) => {
    const messageText = text || inputValue.trim();
    if (messageText && !isLoading) {
      setShowSendPulse(true);
      setTimeout(() => setShowSendPulse(false), 300);
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

  // Premium floating launcher button
  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 group"
        aria-label="Open engineering assistant"
      >
        {/* Animated pulse rings */}
        <div className="absolute inset-0 rounded-full">
          <div className="absolute inset-0 rounded-full bg-primary/20 animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite]" />
          <div className="absolute inset-[-4px] rounded-full bg-primary/10 animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite_0.5s]" />
        </div>
        
        {/* Multi-layer glow */}
        <div className="absolute inset-[-8px] rounded-full bg-gradient-to-br from-primary/40 to-primary/20 blur-xl 
          opacity-60 group-hover:opacity-100 group-hover:scale-110 transition-all duration-500" />
        <div className="absolute inset-[-4px] rounded-full bg-primary/30 blur-md 
          group-hover:bg-primary/50 transition-all duration-300" />
        
        {/* Main button with shimmer */}
        <div className="relative h-16 w-16 rounded-full overflow-hidden
          shadow-[0_8px_32px_rgba(0,0,0,0.3),0_4px_16px_rgba(var(--primary),0.4),inset_0_1px_0_rgba(255,255,255,0.2)]
          group-hover:shadow-[0_12px_48px_rgba(0,0,0,0.4),0_8px_24px_rgba(var(--primary),0.5),inset_0_1px_0_rgba(255,255,255,0.3)]
          transition-all duration-300 ease-out
          group-hover:scale-110 group-active:scale-95">
          
          {/* Gradient background */}
          <div className="absolute inset-0 bg-gradient-to-br from-primary via-primary to-[hsl(355,75%,35%)]" />
          
          {/* Shimmer effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent 
            translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
          
          {/* Icon */}
          <div className="relative h-full w-full flex items-center justify-center">
            <MessageCircle className="h-7 w-7 text-white drop-shadow-lg
              transition-all duration-300 group-hover:scale-110 group-hover:rotate-[-8deg]" />
          </div>
        </div>
        
        {/* Status indicator with glow */}
        <span className="absolute top-0 right-0 h-4 w-4">
          <span className="absolute inset-0 rounded-full bg-emerald-400 animate-ping opacity-75" />
          <span className="absolute inset-0 rounded-full bg-emerald-400 border-2 border-white
            shadow-[0_0_8px_rgba(52,211,153,0.6)]" />
        </span>
        
        {/* Premium tooltip */}
        <div className="absolute bottom-full right-0 mb-3 
          opacity-0 group-hover:opacity-100 translate-y-2 group-hover:translate-y-0
          transition-all duration-300 pointer-events-none">
          <div className="relative px-4 py-2 rounded-xl
            bg-foreground backdrop-blur-xl
            shadow-2xl shadow-black/20">
            <p className="text-background font-medium text-sm whitespace-nowrap">Vectis Assistant</p>
            <p className="text-background/60 text-xs flex items-center gap-1.5 mt-0.5">
              <Sparkles className="h-3 w-3" /> AI-Powered
            </p>
            {/* Arrow */}
            <div className="absolute -bottom-1.5 right-6 w-3 h-3 rotate-45 bg-foreground" />
          </div>
        </div>
      </button>
    );
  }

  // Minimized state - light theme
  if (isMinimized) {
    return (
      <div className="fixed bottom-6 right-6 z-50 w-80
        rounded-2xl overflow-hidden
        bg-background/95 backdrop-blur-2xl
        shadow-[0_8px_32px_-8px_rgba(0,0,0,0.15),0_4px_16px_-4px_rgba(0,0,0,0.1)]
        border border-border/50
        animate-in slide-in-from-bottom-4 fade-in duration-300">
        
        <div className="relative p-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Logo avatar */}
            <div className="relative">
              <div className="h-10 w-10 rounded-xl 
                bg-muted border border-border/50
                flex items-center justify-center 
                shadow-sm
                animate-[float_3s_ease-in-out_infinite]">
                <img src={logo} alt="Vectis" className="h-6 w-6 object-contain" />
              </div>
              <span className="absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full 
                bg-emerald-500 border-2 border-background
                shadow-[0_0_6px_rgba(52,211,153,0.6)]" />
            </div>
            <div>
              <span className="font-semibold text-foreground text-sm tracking-tight">Vectis Assistant</span>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 shadow-[0_0_4px_rgba(52,211,153,0.8)]" />
                <span className="text-[11px] text-muted-foreground font-medium">Online • Ready to help</span>
              </div>
            </div>
          </div>
          <div className="flex gap-0.5">
            <button
              onClick={() => setIsMinimized(false)}
              className="h-8 w-8 rounded-lg flex items-center justify-center
                text-muted-foreground hover:text-foreground hover:bg-muted 
                active:bg-muted/80 transition-all duration-200"
            >
              <ChevronDown className="h-4 w-4 rotate-180 transition-transform" />
            </button>
            <button
              onClick={() => setIsOpen(false)}
              className="h-8 w-8 rounded-lg flex items-center justify-center
                text-muted-foreground hover:text-foreground hover:bg-muted 
                active:bg-muted/80 transition-all duration-200"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Full chat panel - Light theme with clean design
  return (
    <div className="fixed bottom-6 right-6 z-50 w-[420px] h-[620px]
      flex flex-col rounded-3xl overflow-hidden
      bg-background
      shadow-[0_25px_60px_-15px_rgba(0,0,0,0.2),0_10px_30px_-10px_rgba(0,0,0,0.1)]
      border border-border/40
      animate-in slide-in-from-bottom-6 fade-in zoom-in-95 duration-400">
      
      {/* Header - Light sophisticated gradient */}
      <div className="relative shrink-0 overflow-hidden">
        {/* Light gradient background */}
        <div className="absolute inset-0 bg-gradient-to-br from-muted via-muted/80 to-background" />
        
        <div className="relative p-5 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Logo avatar */}
            <div className="relative">
              <div className="relative h-12 w-12 rounded-2xl 
                bg-background border border-border/50
                flex items-center justify-center 
                shadow-md
                animate-[float_4s_ease-in-out_infinite]">
                <img src={logo} alt="Vectis" className="h-8 w-8 object-contain" />
              </div>
              
              {/* Status indicator */}
              <span className="absolute -bottom-0.5 -right-0.5 h-3.5 w-3.5 rounded-full 
                bg-emerald-500 border-[2.5px] border-background
                shadow-[0_0_8px_rgba(52,211,153,0.5)]" />
            </div>
            
            <div>
              <h3 className="font-semibold text-foreground text-[15px] tracking-tight">
                Vectis Assistant
              </h3>
              {partContext?.name && (
                <p className="text-[12px] text-muted-foreground mt-0.5 font-medium">
                  <span className="text-primary">Viewing: {partContext.name}</span>
                </p>
              )}
            </div>
          </div>
          
          {/* Control buttons */}
          <div className="flex gap-1">
            <button
              onClick={clearHistory}
              className="h-9 w-9 rounded-xl flex items-center justify-center
                text-muted-foreground hover:text-foreground hover:bg-background 
                active:bg-muted transition-all duration-200"
              title="Clear conversation"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
            <button
              onClick={() => setIsMinimized(true)}
              className="h-9 w-9 rounded-xl flex items-center justify-center
                text-muted-foreground hover:text-foreground hover:bg-background 
                active:bg-muted transition-all duration-200"
            >
              <ChevronDown className="h-4 w-4" />
            </button>
            <button
              onClick={() => setIsOpen(false)}
              className="h-9 w-9 rounded-xl flex items-center justify-center
                text-muted-foreground hover:text-foreground hover:bg-background 
                active:bg-muted transition-all duration-200"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        
        {/* Bottom border */}
        <div className="absolute bottom-0 left-0 right-0 h-px bg-border/50" />
      </div>

      {/* Messages area - Light background with subtle gradient */}
      <div className="flex-1 overflow-hidden relative bg-gradient-to-b from-muted/30 to-background">
        <ScrollArea className="h-full [&_[data-radix-scroll-area-scrollbar]]:hidden">
          <div ref={scrollRef} className="p-5 space-y-5">
            {messages.length === 0 ? (
              <EmptyState 
                onPromptClick={handleSend} 
                partContext={partContext}
              />
            ) : (
              <>
                {messages.map((message, index) => (
                  <MessageBubble key={index} message={message} index={index} />
                ))}
                {isLoading && messages[messages.length - 1]?.role === 'user' && (
                  <TypingIndicator />
                )}
              </>
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Input area - Light glass design */}
      <div className="relative shrink-0 p-5 pt-3 bg-background border-t border-border/50">
        <div className="relative">
          {/* Input container */}
          <div className={cn(
            "relative rounded-2xl overflow-hidden transition-all duration-300",
            "bg-muted/50 border border-border/50",
            "shadow-sm",
            inputRef.current === document.activeElement && "border-primary/40 shadow-[0_0_0_3px_hsl(var(--primary)/0.1)]"
          )}>
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about materials, heat treatment, tolerances..."
              disabled={isLoading}
              rows={1}
              className="w-full resize-none bg-transparent
                pl-5 pr-14 py-4
                text-sm text-foreground placeholder:text-muted-foreground
                focus:outline-none disabled:opacity-40
                transition-all duration-200"
              style={{ maxHeight: '120px' }}
            />
            
            {/* Send button */}
            <div className="absolute right-2 bottom-2">
              <button
                onClick={() => handleSend()}
                disabled={!inputValue.trim() || isLoading}
                className={cn(
                  "h-10 w-10 rounded-xl flex items-center justify-center",
                  "bg-gradient-to-br from-primary via-primary to-[hsl(355,70%,38%)]",
                  "shadow-[0_4px_12px_hsl(var(--primary)/0.3)]",
                  "disabled:opacity-30 disabled:shadow-none disabled:cursor-not-allowed",
                  "hover:shadow-[0_6px_20px_hsl(var(--primary)/0.4)]",
                  "hover:scale-105 active:scale-95",
                  "transition-all duration-200",
                  showSendPulse && "animate-[sendPulse_0.3s_ease-out]"
                )}
              >
                {isLoading ? (
                  <Loader2 className="h-5 w-5 text-white animate-spin" />
                ) : (
                  <Send className="h-5 w-5 text-white" />
                )}
              </button>
            </div>
          </div>
          
          {/* Footer text */}
          <p className="text-[10px] text-muted-foreground text-center mt-3 font-medium tracking-wide">
            POWERED BY AI • PRESS ENTER TO SEND
          </p>
        </div>
      </div>
    </div>
  );
}

// Premium Empty State - Light theme
function EmptyState({ 
  onPromptClick, 
  partContext 
}: { 
  onPromptClick: (text: string) => void;
  partContext?: PartContext;
}) {
  return (
    <div className="py-6">
      {/* Welcome card - Light theme */}
      <div className="relative mb-8 animate-in fade-in-0 slide-in-from-bottom-3 duration-500">
        <div className="relative rounded-2xl overflow-hidden
          bg-background
          border border-border/50
          shadow-sm">
          
          <div className="p-5">
            <div className="flex items-start gap-4">
              {/* Logo avatar */}
              <div className="relative shrink-0">
                <div className="h-11 w-11 rounded-xl 
                  bg-muted border border-border/50
                  flex items-center justify-center
                  shadow-sm
                  animate-[float_3s_ease-in-out_infinite]">
                  <img src={logo} alt="Vectis" className="h-7 w-7 object-contain" />
                </div>
              </div>
              
              <div className="flex-1 pt-0.5">
                <p className="text-[14px] text-foreground/80 leading-relaxed">
                  Hey there! I'm your <span className="text-primary font-medium">engineering assistant</span>. 
                  I specialize in heat treatment, stock sizes, HRC values, and material calculations.
                </p>
                {partContext?.name && (
                  <p className="text-[13px] text-primary mt-3 font-medium flex items-center gap-2
                    px-3 py-2 rounded-lg bg-primary/10 border border-primary/20 w-fit">
                    <Sparkles className="h-3.5 w-3.5" />
                    Working with {partContext.name}
                    {partContext.material && ` in ${partContext.material}`}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick prompts - Light theme cards */}
      <div className="space-y-3">
        <p className="text-[11px] text-muted-foreground font-semibold tracking-wider uppercase px-1">
          Suggested questions
        </p>
        <div className="grid gap-2.5">
          {quickPrompts.map((item, i) => {
            const Icon = item.icon;
            return (
              <button
                key={i}
                onClick={() => onPromptClick(item.prompt)}
                className="group relative w-full text-left rounded-xl overflow-hidden
                  animate-in fade-in-0 slide-in-from-bottom-2 duration-300"
                style={{ animationDelay: `${i * 100}ms` }}
              >
                <div className="relative flex items-center gap-3 px-4 py-3.5
                  bg-background border border-border/50
                  group-hover:border-primary/30 group-hover:bg-muted/30
                  rounded-xl transition-all duration-300
                  shadow-sm group-hover:shadow-md">
                  
                  {/* Icon container */}
                  <div className="h-9 w-9 rounded-lg shrink-0
                    bg-primary/10 border border-primary/20
                    flex items-center justify-center
                    group-hover:bg-primary/15
                    transition-all duration-300">
                    <Icon className="h-4 w-4 text-primary/80 group-hover:text-primary transition-colors" />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <p className="text-[11px] text-muted-foreground font-medium mb-0.5 group-hover:text-foreground/60 transition-colors">
                      {item.title}
                    </p>
                    <p className="text-[13px] text-foreground/70 group-hover:text-foreground truncate transition-colors">
                      {item.prompt}
                    </p>
                  </div>
                  
                  {/* Arrow indicator */}
                  <div className="h-6 w-6 rounded-full 
                    bg-muted group-hover:bg-primary/20
                    flex items-center justify-center shrink-0
                    opacity-0 group-hover:opacity-100 
                    translate-x-2 group-hover:translate-x-0
                    transition-all duration-300">
                    <Send className="h-3 w-3 text-muted-foreground group-hover:text-primary" />
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// Premium Typing Indicator - Light theme
function TypingIndicator() {
  return (
    <div className="flex gap-3.5 animate-in fade-in-0 slide-in-from-bottom-2 duration-300">
      <div className="h-10 w-10 rounded-xl shrink-0
        bg-muted border border-border/50
        flex items-center justify-center
        shadow-sm
        animate-[float_2s_ease-in-out_infinite]">
        <img src={logo} alt="Vectis" className="h-6 w-6 object-contain" />
      </div>
      
      <div className="rounded-2xl rounded-tl-md px-5 py-4
        bg-muted border border-border/50 shadow-sm">
        <div className="flex items-center gap-2">
          <div className="flex gap-1">
            {[0, 1, 2].map((i) => (
              <span 
                key={i}
                className="h-2 w-2 rounded-full bg-primary/60"
                style={{ 
                  animation: 'elasticBounce 1.4s ease-in-out infinite',
                  animationDelay: `${i * 0.16}s`
                }} 
              />
            ))}
          </div>
          <span className="text-[11px] text-muted-foreground font-medium ml-2">Thinking...</span>
        </div>
      </div>
    </div>
  );
}

// Premium Message Bubble - Light theme
function MessageBubble({ message, index }: { message: Message; index: number }) {
  const isUser = message.role === 'user';
  const displayContent = isUser ? message.content : cleanMarkdown(message.content);
  
  return (
    <div 
      className={cn(
        'flex gap-3.5',
        'animate-in fade-in-0 duration-400',
        isUser ? 'flex-row-reverse slide-in-from-right-3' : 'flex-row slide-in-from-left-3'
      )}
      style={{ animationDelay: `${index * 50}ms` }}
    >
      {/* Avatar */}
      <div className={cn(
        'h-10 w-10 rounded-xl shrink-0 flex items-center justify-center',
        isUser 
          ? 'bg-muted border border-border/50'
          : 'bg-background border border-border/50 shadow-sm'
      )}>
        {isUser ? (
          <User className="h-5 w-5 text-muted-foreground" />
        ) : (
          <img src={logo} alt="Vectis" className="h-6 w-6 object-contain" />
        )}
      </div>

      {/* Message bubble */}
      <div
        className={cn(
          'max-w-[78%] rounded-2xl px-4 py-3.5 text-[14px] leading-relaxed',
          isUser
            ? 'bg-gradient-to-br from-primary via-primary to-[hsl(355,68%,40%)] text-white rounded-tr-md shadow-[0_4px_16px_hsl(var(--primary)/0.3)]'
            : 'bg-muted text-foreground rounded-tl-md border border-border/50'
        )}
      >
        <div className="whitespace-pre-wrap break-words">
          {displayContent || (
            <span className="opacity-40 italic">...</span>
          )}
        </div>
      </div>
    </div>
  );
}
