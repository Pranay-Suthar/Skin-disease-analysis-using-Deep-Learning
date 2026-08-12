import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X } from 'lucide-react';
import { Icon } from '@iconify/react';

const NAV_LINKS = [
  { to: '/',          label: 'Home' },
  { to: '/analyze',   label: 'Analyze' },
  { to: '/resources', label: 'Resources' },
  { to: '/hospitals', label: 'Find Care' },
  { to: '/about',     label: 'About' },
];

interface HeaderProps {
  showAnalyzeButton?: boolean;
  analyzeButtonText?: string;
}

export default function Header({ showAnalyzeButton = true, analyzeButtonText = 'Start Analysis' }: HeaderProps) {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();

  return (
    <header className="fixed top-0 left-0 right-0 z-50">
      <div className="mx-auto max-w-7xl px-4 pt-3">
        <div className="glass rounded-2xl px-6 py-3 flex items-center justify-between">

          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center shadow-sm"
                 style={{ background: 'linear-gradient(135deg,#e11d48,#f59e0b)' }}>
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5 text-white">
                <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/>
              </svg>
            </div>
            <span className="font-bold text-lg text-gray-900 group-hover:opacity-80 transition-opacity">
              Derma<span style={{ color: '#e11d48' }}>AI</span>
            </span>
          </Link>

          {/* Desktop Nav */}
          <nav className="hidden lg:flex items-center gap-1">
            {NAV_LINKS.map(item => (
              <Link
                key={item.to}
                to={item.to}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  pathname === item.to
                    ? 'text-rose-700 bg-rose-50'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>

          {/* Right */}
          <div className="flex items-center gap-2">
            <a 
              href="https://github.com/Pranay-Suthar/Skin-disease-analysis-using-Deep-Learning"
              target="_blank"
              rel="noopener noreferrer"
              title="View Source on GitHub"
              className="p-2 text-gray-500 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors mr-1 sm:mr-2"
            >
              <Icon icon="mdi:github" className="w-5 h-5" />
            </a>
            {showAnalyzeButton && (
              <Link
                to="/analyze"
                className="hidden sm:inline-flex btn-primary text-sm px-5 py-2"
              >
                {analyzeButtonText}
              </Link>
            )}
            <button
              onClick={() => setOpen(!open)}
              className="lg:hidden p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition"
            >
              {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {open && (
          <div className="glass mt-2 rounded-2xl p-4 animate-slide-down">
            {NAV_LINKS.map(item => (
              <Link
                key={item.to}
                to={item.to}
                onClick={() => setOpen(false)}
                className="block px-4 py-3 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-50 transition"
              >
                {item.label}
              </Link>
            ))}
            {showAnalyzeButton && (
              <Link
                to="/analyze"
                onClick={() => setOpen(false)}
                className="btn-primary w-full justify-center mt-3"
              >
                {analyzeButtonText}
              </Link>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
