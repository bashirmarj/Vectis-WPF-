import React from 'react';
import { useScrollAnimation } from '@/hooks/useScrollAnimation';
import { cn } from '@/lib/utils';

const steps = [
  { id: '01', title: 'Upload', description: 'Submit your CAD files' },
  { id: '02', title: 'Quote', description: '24-hour turnaround' },
  { id: '03', title: 'Manufacture', description: 'Precision production' },
  { id: '04', title: 'Quality', description: 'Rigorous inspection' },
  { id: '05', title: 'Deliver', description: 'On-time shipping' },
];

interface ProcessSectionGeminiProps {
  className?: string;
}

const ProcessSectionGemini: React.FC<ProcessSectionGeminiProps> = ({ className }) => {
  const { ref, isVisible } = useScrollAnimation();

  return (
    <div ref={ref} className={cn('w-full', className)}>
      {/* Desktop: Horizontal layout */}
      <div className="hidden md:grid md:grid-cols-5 gap-4">
        {steps.map((step, index) => (
          <div
            key={step.id}
            className={cn(
              'group relative bg-black/40 backdrop-blur-sm border border-white/10 rounded-lg p-6 transition-all duration-500 hover:border-primary/50 hover:bg-black/60 overflow-hidden',
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
            )}
            style={{ transitionDelay: `${index * 100}ms` }}
          >
            {/* Large number background */}
            <span className="absolute -top-4 -right-2 text-[5rem] font-bold text-white/5 leading-none select-none group-hover:text-primary/10 transition-colors duration-500">
              {step.id}
            </span>
            
            {/* Red bar animation on hover */}
            <div className="absolute bottom-0 left-0 w-full h-1 bg-primary scale-x-0 group-hover:scale-x-100 transition-transform duration-500 origin-left" />
            
            {/* Content */}
            <div className="relative z-10">
              <span className="text-xs font-mono text-primary/80 tracking-wider mb-2 block">
                STEP {step.id}
              </span>
              <h4 className="text-lg font-bold text-white mb-2 group-hover:text-primary transition-colors duration-300">
                {step.title}
              </h4>
              <p className="text-sm text-gray-400 group-hover:text-gray-300 transition-colors duration-300">
                {step.description}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Mobile: 2-column grid */}
      <div className="grid grid-cols-2 gap-4 md:hidden">
        {steps.map((step, index) => (
          <div
            key={step.id}
            className={cn(
              'group relative bg-black/40 backdrop-blur-sm border border-white/10 rounded-lg p-5 transition-all duration-500 hover:border-primary/50 overflow-hidden',
              index === steps.length - 1 ? 'col-span-2' : '',
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
            )}
            style={{ transitionDelay: `${index * 100}ms` }}
          >
            {/* Large number background */}
            <span className="absolute -top-2 -right-1 text-[3rem] font-bold text-white/5 leading-none select-none">
              {step.id}
            </span>
            
            {/* Red bar animation */}
            <div className="absolute bottom-0 left-0 w-full h-0.5 bg-primary scale-x-0 group-hover:scale-x-100 transition-transform duration-500 origin-left" />
            
            {/* Content */}
            <div className="relative z-10">
              <span className="text-[10px] font-mono text-primary/80 tracking-wider mb-1 block">
                STEP {step.id}
              </span>
              <h4 className="text-base font-bold text-white mb-1">
                {step.title}
              </h4>
              <p className="text-xs text-gray-400">
                {step.description}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProcessSectionGemini;
