import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import Header from '../components/Header';
import { Activity, Upload, Zap, Shield, Users, MapPin, BookOpen, MessageCircle, Heart, CheckCircle2, ArrowRight, X, LogIn, UserPlus } from 'lucide-react';

const AuthModal = ({ isOpen, onClose }: { isOpen: boolean, onClose: () => void }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({ name: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    const endpoint = isLogin ? '/api/login/' : '/api/signup/';
    try {
      const response = await fetch(`http://127.0.0.1:5000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      
      if (response.ok) {
        setSuccess(isLogin ? 'Login successful!' : 'Account created successfully!');
        if (data.user) {
          localStorage.setItem('user', JSON.stringify(data.user));
        }
        setTimeout(() => {
          onClose();
          window.location.reload();
        }, 1500);
      } else {
        setError(data.error || 'Authentication failed');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md overflow-hidden relative animate-fade-in">
        <button onClick={onClose} className="absolute top-4 right-4 text-gray-500 hover:text-gray-800 transition-colors">
          <X className="w-5 h-5" />
        </button>
        
        <div className="p-8">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
            <p className="text-gray-500">{isLogin ? 'Sign in to access your analysis history.' : 'Join us for personalized skin insights.'}</p>
          </div>

          <div className="flex bg-gray-100 rounded-lg p-1 mb-6">
            <button 
              onClick={() => setIsLogin(true)}
              className={`flex-1 py-2 text-sm font-semibold rounded-md transition-all ${isLogin ? 'bg-white shadow text-teal-600' : 'text-gray-500 hover:text-gray-700'}`}
            >
              Log In
            </button>
            <button 
              onClick={() => setIsLogin(false)}
              className={`flex-1 py-2 text-sm font-semibold rounded-md transition-all ${!isLogin ? 'bg-white shadow text-teal-600' : 'text-gray-500 hover:text-gray-700'}`}
            >
              Sign Up
            </button>
          </div>

          {error && <div className="mb-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg">{error}</div>}
          {success && <div className="mb-4 p-3 bg-green-50 text-green-600 text-sm rounded-lg">{success}</div>}

          <form onSubmit={handleSubmit} className="space-y-4">
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input 
                  type="text" required 
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500 outline-none transition-all"
                  value={formData.name} onChange={e => setFormData({...formData, name: e.target.value})}
                />
              </div>
            )}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
              <input 
                type="email" required 
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500 outline-none transition-all"
                value={formData.email} onChange={e => setFormData({...formData, email: e.target.value})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
              <input 
                type="password" required 
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500 outline-none transition-all"
                value={formData.password} onChange={e => setFormData({...formData, password: e.target.value})}
              />
            </div>
            
            <button 
              type="submit" disabled={loading}
              className="w-full py-3 mt-6 bg-gradient-to-r from-teal-600 to-emerald-600 text-white font-bold rounded-lg hover:from-teal-700 hover:to-emerald-700 transition-all shadow-md flex justify-center items-center gap-2"
            >
              {loading ? 'Please wait...' : isLogin ? <><LogIn className="w-5 h-5"/> Sign In</> : <><UserPlus className="w-5 h-5"/> Sign Up</>}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default function Home() {
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const user = localStorage.getItem('user');
  
  return (
    <div className="min-h-screen bg-white pt-20">
      <Header />
      <AuthModal isOpen={isAuthOpen} onClose={() => setIsAuthOpen(false)} />

      {/* Hero Section */}
      <section className="px-5 py-8 md:py-14 relative overflow-hidden">
        {/* Soft background blobs */}
        <div className="absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-96 h-96 rounded-full opacity-30" style={{ background: 'radial-gradient(circle,#ede9fe,transparent)' }} />
          <div className="absolute -bottom-20 -left-20 w-80 h-80 rounded-full opacity-20" style={{ background: 'radial-gradient(circle,#ecfeff,transparent)' }} />
        </div>

        <div className="mx-auto max-w-7xl">
          <div className="grid gap-12 lg:grid-cols-2 lg:gap-12 items-center">
            <div className="animate-slide-up">
              <div className="inline-flex items-center gap-2 mb-5 px-4 py-1.5 rounded-full text-sm font-semibold" style={{ background:'#ede9fe', color:'#7c3aed' }}>
                <Zap className="w-4 h-4" /> AI-Powered Skin Analysis
              </div>
              <h1 className="section-title mb-6">
                Understanding Your Skin, Made Simple
              </h1>
              <p className="text-xl text-gray-600 mb-8 leading-relaxed font-light">
                Get instant, accurate insights about your skin health using cutting-edge AI. From diagnosis to personalized care recommendations—all from your phone.
              </p>
              <div className="flex flex-wrap gap-4 mb-8">
                <Link to="/analyze" className="btn-primary group">
                  <span className="flex items-center gap-2">
                    <span>Start Free Analysis</span>
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </span>
                </Link>
                {!user ? (
                  <button onClick={() => setIsAuthOpen(true)} className="btn-secondary flex items-center gap-2">
                    <UserPlus className="w-4 h-4" /> Sign In / Join
                  </button>
                ) : (
                  <button onClick={() => { localStorage.removeItem('user'); window.location.reload(); }} className="btn-secondary flex items-center gap-2">
                    Logout
                  </button>
                )}
              </div>
              <div className="flex items-center gap-6 text-sm">
                <div>
                  <p className="font-bold text-gray-900">95%+</p>
                  <p className="text-gray-600">Accuracy</p>
                </div>
                <div className="w-px h-12 bg-gray-300"></div>
                <div>
                  <p className="font-bold text-gray-900">50K+</p>
                  <p className="text-gray-600">Users</p>
                </div>
                <div className="w-px h-12 bg-gray-300"></div>
                <div>
                  <p className="font-bold text-gray-900">180+</p>
                  <p className="text-gray-600">Clinics</p>
                </div>
              </div>
            </div>

            {/* Hero Image */}
            <div className="relative animate-slide-down">
              <div className="absolute -inset-6 rounded-3xl blur-2xl opacity-50" style={{ background: 'linear-gradient(135deg,#ede9fe,#ecfeff)' }}></div>
              <div className="relative rounded-3xl overflow-hidden shadow-xl border border-gray-100">
                <img
                  src="https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=600&h=600&fit=crop&q=80"
                  alt="Skincare"
                  className="w-full h-full object-cover"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-14 px-5" style={{ background: 'linear-gradient(135deg,#7c3aed,#6d28d9)' }}>
        <div className="mx-auto max-w-7xl">
          <div className="grid gap-8 md:grid-cols-3">
            {[
              { value: '95%+', label: 'Accuracy Rate', Icon: Activity },
              { value: '50K+', label: 'Active Users', Icon: Users },
              { value: '180+', label: 'Partner Clinics', Icon: Heart },
            ].map((stat, idx) => (
              <div key={idx} className="text-center text-white" style={{ animationDelay: `${idx * 0.1}s` }}>
                <div className="flex justify-center mb-3">
                  <stat.Icon className="w-9 h-9 opacity-90" />
                </div>
                <div className="text-5xl font-bold mb-1">{stat.value}</div>
                <div className="text-white/80 text-sm font-medium">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="px-5 py-24 relative overflow-hidden">
        <div className="mx-auto max-w-7xl">
          <div className="text-center mb-20">
            <h2 className="section-title mb-4">How It Works</h2>
            <p className="section-subtitle max-w-2xl mx-auto">
              Simple, secure, and scientifically-backed skin analysis in three steps
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-3 relative">
            <div className="hidden md:block absolute top-1/4 left-1/3 right-1/3 h-px" style={{ background: 'linear-gradient(90deg,#ede9fe,#ecfeff)' }}></div>

            {[
              {
                step: 1,
                Icon: Upload,
                title: 'Upload Image',
                description: 'Take a clear photo of your skin concern and upload it securely to our platform.',
              },
              {
                step: 2,
                Icon: Activity,
                title: 'AI Analysis',
                description: 'Our advanced AI instantly analyzes your image with 95%+ accuracy using deep learning.',
              },
              {
                step: 3,
                Icon: CheckCircle2,
                title: 'Get Guidance',
                description: 'Receive personalized recommendations and connect with local dermatologists.',
              },
            ].map((feature, idx) => (
              <div key={idx} className="card group hover:-translate-y-2 transition-all duration-300">
                <div className="flex items-center justify-between mb-4">
                  <div className="p-3 rounded-xl" style={{ background:'#ede9fe' }}>
                    <feature.Icon className="w-7 h-7" style={{ color:'#7c3aed' }} />
                  </div>
                  <div className="w-9 h-9 rounded-full text-white flex items-center justify-center font-bold text-sm" style={{ background:'linear-gradient(135deg,#7c3aed,#06b6d4)' }}>
                    {feature.step}
                  </div>
                </div>
                <h3 className="text-lg font-bold mb-2 text-gray-900">{feature.title}</h3>
                <p className="text-gray-500 text-sm leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Feature Highlight */}
      <section className="bg-gray-50 px-5 py-24">
        <div className="mx-auto max-w-7xl">
          <div className="grid gap-16 lg:grid-cols-2 items-center">
            <div className="relative group">
              <div className="absolute -inset-4 rounded-3xl blur-2xl opacity-40 group-hover:opacity-70 transition-opacity" style={{ background:'linear-gradient(135deg,#ede9fe,#ecfeff)' }}></div>
              <div className="relative rounded-2xl overflow-hidden shadow-lg border border-gray-100">
                <img
                  src="https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=500&h=500&fit=crop&q=80"
                  alt="Real-time analysis"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                />
              </div>
            </div>
            <div className="animate-slide-up">
              <h2 className="text-4xl font-bold mb-6 text-gray-900 font-display">Real-Time Analysis</h2>
              <p className="text-gray-500 mb-8 leading-relaxed">
                Get instant results with detailed insights about your skin condition. Our AI uses cutting-edge computer vision to identify patterns and provide personalized recommendations.
              </p>
              <div className="space-y-4">
                {[
                  { Icon: Zap, title: 'Instant Results', desc: 'Analysis in seconds' },
                  { Icon: Shield, title: 'Privacy First', desc: 'Your data is encrypted' },
                  { Icon: Heart, title: 'Expert Guidance', desc: 'Backed by dermatologists' },
                ].map((item, idx) => (
                  <div key={idx} className="flex gap-4 items-start">
                    <div className="p-2 rounded-lg flex-shrink-0" style={{ background:'#ede9fe' }}>
                      <item.Icon className="w-5 h-5" style={{ color:'#7c3aed' }} />
                    </div>
                    <div>
                      <p className="font-semibold text-gray-900">{item.title}</p>
                      <p className="text-sm text-gray-500">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Quick Links */}
      <section className="px-5 py-24">
        <div className="mx-auto max-w-7xl">
          <div className="text-center mb-16">
            <h2 className="section-title mb-4">Everything You Need</h2>
            <p className="section-subtitle max-w-2xl mx-auto">
              Access resources, find care, and get personalized support
            </p>
          </div>

          <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-4">
            {[
              { to: '/analyze',   Icon: Activity,       title: 'Analyze',      desc: 'AI-powered skin analysis' },
              { to: '/resources', Icon: BookOpen,        title: 'Learn',        desc: 'Educational resources' },
              { to: '/hospitals', Icon: MapPin,          title: 'Find Care',    desc: 'Dermatology clinics nearby' },
              { to: '/skinbot',   Icon: MessageCircle,   title: 'Chat Support', desc: 'Get personalized help' },
            ].map((link, idx) => (
              <Link
                key={idx}
                to={link.to}
                className="card group hover:scale-105 hover:-translate-y-1 transition-all duration-300 cursor-pointer"
                style={{ borderColor: 'transparent' }}
                onMouseEnter={e => (e.currentTarget.style.borderColor = '#7c3aed')}
                onMouseLeave={e => (e.currentTarget.style.borderColor = 'transparent')}
              >
                <div className="p-2.5 rounded-xl mb-4 inline-flex" style={{ background:'#ede9fe' }}>
                  <link.Icon className="w-6 h-6" style={{ color:'#7c3aed' }} />
                </div>
                <h3 className="text-base font-bold mb-1 text-gray-900">{link.title}</h3>
                <p className="text-gray-500 text-sm">{link.desc}</p>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="px-5 py-24 relative overflow-hidden" style={{ background: 'linear-gradient(135deg,#7c3aed 0%,#6d28d9 50%,#06b6d4 100%)' }}>
        <div className="mx-auto max-w-2xl text-center relative z-10">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6 font-display">Ready to Transform Your Skin Care?</h2>
          <p className="text-white/80 mb-10 text-lg leading-relaxed">
            Join thousands of users getting personalized skin insights powered by AI.
          </p>
          <Link
            to="/analyze"
            className="inline-flex items-center gap-2 px-10 py-4 bg-white font-bold rounded-xl hover:shadow-2xl hover:scale-105 transition-all duration-300 active:scale-95"
            style={{ color:'#7c3aed' }}
          >
            <span>Get Started Free</span>
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ background:'#0f0a1a' }}>
        <div className="px-5 py-20">
          <div className="mx-auto max-w-7xl">
            <div className="grid gap-12 md:grid-cols-4 mb-16">
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background:'linear-gradient(135deg,#7c3aed,#06b6d4)' }}>
                    <Activity className="w-5 h-5 text-white" />
                  </div>
                  <span className="font-bold text-lg text-white">DermaAI</span>
                </div>
                <p className="text-white/60 text-sm leading-relaxed">
                  Empowering users with AI-driven skin care insights backed by dermatological expertise.
                </p>
              </div>
              {[
                {
                  title: 'Product',
                  links: [
                    { label: 'Analyze', to: '/analyze' },
                    { label: 'Resources', to: '/resources' },
                    { label: 'Find Care', to: '/hospitals' },
                  ],
                },
                {
                  title: 'Company',
                  links: [
                    { label: 'About', to: '/about' },
                    { label: 'Contact', to: '/contact' },
                    { label: 'Blog', to: '#' },
                  ],
                },
                {
                  title: 'Legal',
                  links: [
                    { label: 'Privacy Policy', to: '#' },
                    { label: 'Terms of Service', to: '#' },
                    { label: 'Disclaimer', to: '#' },
                  ],
                },
              ].map((section, idx) => (
                <div key={idx}>
                  <h3 className="font-bold mb-4 text-white">{section.title}</h3>
                  <ul className="space-y-3">
                    {section.links.map((link, linkIdx) => (
                      <li key={linkIdx}>
                        <Link to={link.to} className="text-white/60 hover:text-teal-300 transition-colors text-sm font-medium">
                          {link.label}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            {/* Divider */}
            <div className="h-px bg-gradient-to-r from-transparent via-white/20 to-transparent mb-8"></div>

            {/* Bottom Footer */}
            <div className="text-center">
              <p className="text-sm text-white/50">© 2026 DermaAI. All rights reserved. | Educational tool - not a medical diagnosis device.</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
