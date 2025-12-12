import React from 'react';
import { Upload, FileText, Cog, CheckCircle, Truck } from 'lucide-react';
import { useScrollAnimation, useStaggerAnimation } from '@/hooks/useScrollAnimation';
import { cn } from '@/lib/utils';

const steps = [
  { icon: Upload, label: 'Upload', description: 'Submit your CAD files' },
  { icon: FileText, label: 'Quote', description: '24-hour turnaround' },
  { icon: Cog, label: 'Manufacture', description: 'Precision production' },
  { icon: CheckCircle, label: 'Quality', description: 'Rigorous inspection' },
  { icon: Truck, label: 'Deliver', description: 'On-time shipping' },
];

const ProcessTimeline: React.FC<{ className?: string }> = ({ className }) => {
  const { ref, isVisible, getDelayStyle } = useStaggerAnimation(steps.length, 150);

  return (
    <div ref={ref} className={cn('w-full', className)}>
      <div className="relative">
        {/* Connection line */}
        <div className="absolute top-8 left-[10%] right-[10%] h-0.5 bg-white/20 hidden md:block">
          <div
            className={cn(
              'h-full bg-primary transition-all duration-1000 ease-out',
              isVisible ? 'w-full' : 'w-0'
            )}
            style={{ transitionDelay: '300ms' }}
          />
        </div>

        {/* Steps */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-6 md:gap-4">
          {steps.map((step, index) => {
            const Icon = step.icon;
            return (
              <div
                key={index}
                className={cn(
                  'flex flex-col items-center text-center transition-all duration-500',
                  isVisible
                    ? 'opacity-100 translate-y-0'
                    : 'opacity-0 translate-y-8'
                )}
                style={getDelayStyle(index)}
              >
                <div
                  className={cn(
                    'w-16 h-16 rounded-full bg-neutral-900 border-2 border-primary flex items-center justify-center mb-3 transition-all duration-500 relative z-10',
                    isVisible ? 'scale-100' : 'scale-0'
                  )}
                  style={{ transitionDelay: `${index * 150 + 200}ms` }}
                >
                  <Icon className="w-7 h-7 text-primary" />
                </div>
                <h4 className="font-semibold text-white mb-1">{step.label}</h4>
                <p className="text-xs text-gray-400">{step.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ProcessTimeline;
