import React from 'react';

// Fallback icon component when @iconify/react is not available
export const Icon: React.FC<{ icon: string; className?: string }> = ({
  icon,
  className = '',
}) => {
  // Return a simple SVG icon based on the icon name
  const iconMap: Record<string, string> = {
    'lucide:activity': '⚡',
    'lucide:search': '🔍',
    'lucide:map-pin': '📍',
    'lucide:phone': '☎️',
    'lucide:layers': '📚',
    'lucide:shield': '🛡️',
    'lucide:heart': '❤️',
    'lucide:circle-alert': '⚠️',
    'lucide:play': '▶️',
    'lucide:paperclip': '📎',
    'lucide:send': '📤',
  };

  return <span className={className}>{iconMap[icon] || '●'}</span>;
};

export default Icon;
