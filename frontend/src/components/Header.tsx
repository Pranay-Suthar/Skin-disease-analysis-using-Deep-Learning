import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X } from 'lucide-react';

const NAV_LINKS = [
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
                 style={{ background: 'linear-gradient(135deg,#7c3aed,#06b6d4)' }}>
              <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z"/>
              </svg>
            </div>
            <span className="font-bold text-lg text-gray-900 group-hover:opacity-80 transition-opacity">
              Derma<span style={{ color: '#7c3aed' }}>AI</span>
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
                    ? 'text-violet-700 bg-violet-50'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>

          {/* Right */}
          <div className="flex items-center gap-2">
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
