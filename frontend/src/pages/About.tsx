import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Activity, Users, Globe, ChevronRight, Award, Zap, X } from 'lucide-react';
import Header from '../components/Header';

const fadeIn = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.7, ease: "easeOut" }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.15
    }
  }
};

export default function About() {
  const [isTermsOpen, setIsTermsOpen] = useState(false);

  const stats = [
    { value: '98%', label: 'Diagnostic Accuracy', icon: Activity },
    { value: '2M+', label: 'Analyses Performed', icon: Users },
    { value: '150+', label: 'Countries Reached', icon: Globe },
    { value: '24/7', label: 'AI Availability', icon: Zap },
  ];

  const features = [
    {
      title: 'State-of-the-Art AI',
      description: 'Our proprietary machine learning models are trained on over 5 million clinically validated dermatological images, ensuring unprecedented accuracy.',
      icon: Activity,
      color: 'text-rose-600',
      bg: 'bg-rose-100',
    },
    {
      title: 'Privacy by Design',
      description: 'Bank-level encryption and strict zero-retention policies mean your sensitive health data remains completely private and secure.',
      icon: Shield,
      color: 'text-indigo-600',
      bg: 'bg-indigo-100',
    },
    {
      title: 'Clinically Validated',
      description: 'Developed in collaboration with leading dermatologists and continuously peer-reviewed for medical accuracy and safety.',
      icon: Award,
      color: 'text-amber-600',
      bg: 'bg-amber-100',
    },
  ];

  return (
    <div className="min-h-screen bg-[#fafafa] text-gray-900 overflow-hidden selection:bg-rose-500/30 font-sans pt-28">
      <Header />
      
      {/* Hero Section */}
      <section className="relative pt-12 pb-20 lg:pt-24 lg:pb-32 px-5">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-rose-50 via-[#fafafa] to-[#fafafa] -z-10" />
        
        {/* Abstract Background Shapes */}
        <div className="absolute top-0 left-10 w-72 h-72 bg-rose-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30"></div>
        <div className="absolute top-0 right-10 w-72 h-72 bg-rose-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30"></div>
        <div className="absolute -bottom-8 left-40 w-72 h-72 bg-indigo-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30"></div>

        <div className="max-w-7xl mx-auto relative z-10">
          <motion.div 
            initial="initial"
            animate="animate"
            variants={fadeIn}
            className="max-w-4xl text-center mx-auto"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-rose-50 text-rose-700 font-medium text-sm mb-8 ring-1 ring-rose-200 shadow-sm">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
              </span>
              Empowering Skin Health Globally
            </div>
            <h1 className="text-5xl lg:text-7xl font-extrabold tracking-tight mb-8 text-gray-900 leading-[1.1]">
              Democratizing <br className="hidden sm:block" />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-600 to-rose-600">
                Dermatological Care
              </span>
            </h1>
            <p className="text-xl lg:text-2xl text-gray-600 leading-relaxed max-w-2xl mx-auto">
              We're bridging the gap between advanced artificial intelligence and accessible healthcare, bringing expert-level skin analysis to everyone, everywhere.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 relative z-10">
        <div className="max-w-7xl mx-auto px-5">
          <motion.div 
            variants={staggerContainer}
            initial="initial"
            whileInView="animate"
            viewport={{ once: true, margin: "-100px" }}
            className="grid grid-cols-2 lg:grid-cols-4 gap-6"
          >
            {stats.map((stat, idx) => (
              <motion.div 
                key={idx} 
                variants={fadeIn} 
                className="flex flex-col items-center justify-center p-8 bg-white/60 backdrop-blur-xl rounded-3xl border border-white/20 shadow-xl shadow-rose-900/5 hover:-translate-y-1 transition-transform duration-300"
              >
                <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-rose-500 to-indigo-500 flex items-center justify-center mb-5 text-white shadow-lg shadow-rose-500/30">
                  <stat.icon className="w-6 h-6" />
                </div>
                <div className="text-4xl lg:text-5xl font-bold text-gray-900 mb-2 tracking-tight">{stat.value}</div>
                <div className="text-sm text-gray-500 font-semibold tracking-wide uppercase text-center">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 lg:py-32 px-5 relative bg-gray-50/50">
        <div className="max-w-7xl mx-auto">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16 lg:mb-24"
          >
            <h2 className="text-3xl lg:text-5xl font-extrabold mb-6 text-gray-900">Built for Excellence</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Every aspect of DermaAI is engineered to deliver accurate, reliable, and secure health insights with an uncompromising focus on quality.
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-3 gap-8">
            {features.map((feature, idx) => (
              <motion.div 
                key={idx}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: idx * 0.15, duration: 0.6 }}
                className="group p-8 rounded-[2rem] bg-white border border-gray-100 hover:border-transparent hover:shadow-2xl hover:shadow-rose-900/10 transition-all duration-500 relative overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-gray-50 to-white opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                <div className="relative z-10">
                  <div className={`w-16 h-16 rounded-2xl ${feature.bg} flex items-center justify-center mb-6 group-hover:scale-110 group-hover:rotate-3 transition-transform duration-500`}>
                    <feature.icon className={`w-8 h-8 ${feature.color}`} />
                  </div>
                  <h3 className="text-2xl font-bold mb-4 text-gray-900">{feature.title}</h3>
                  <p className="text-gray-600 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Trust & Safety */}
      <section className="py-24 px-5">
        <div className="max-w-7xl mx-auto">
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 rounded-[3rem] p-8 lg:p-16 relative overflow-hidden shadow-2xl shadow-gray-900/20"
          >
            {/* Background decorative elements */}
            <div className="absolute top-0 right-0 -mr-20 -mt-20 w-96 h-96 rounded-full bg-rose-500/10 blur-3xl"></div>
            <div className="absolute bottom-0 left-0 -ml-20 -mb-20 w-80 h-80 rounded-full bg-rose-500/10 blur-3xl"></div>
            
            <div className="relative z-10 flex flex-col lg:flex-row gap-12 items-center justify-between">
              <div className="max-w-2xl">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-red-500/10 text-red-400 font-semibold text-sm mb-6 border border-red-500/20">
                  <Shield className="w-4 h-4" />
                  Important Notice
                </div>
                <h2 className="text-3xl lg:text-5xl font-bold mb-6 text-white tracking-tight">Medical Disclaimer</h2>
                <p className="text-lg text-gray-300 leading-relaxed mb-8">
                  While DermaAI utilizes advanced artificial intelligence to provide educational insights and preliminary analysis, it is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
                </p>
                <button 
                  onClick={() => setIsTermsOpen(true)}
                  className="inline-flex items-center gap-2 text-white font-bold bg-white/10 hover:bg-white/20 px-6 py-3 rounded-xl transition-colors backdrop-blur-md"
                >
                  Read Full Terms of Service <ChevronRight className="w-5 h-5" />
                </button>
              </div>
              <div className="w-full lg:w-auto flex justify-center">
                <div className="relative">
                  <div className="absolute inset-0 bg-rose-500/20 blur-2xl rounded-full scale-150 animate-pulse"></div>
                  <div className="w-32 h-32 rounded-3xl bg-gradient-to-br from-rose-500 to-indigo-600 flex items-center justify-center relative shadow-2xl rotate-3 hover:rotate-6 transition-transform duration-500">
                    <Shield className="w-16 h-16 text-white" />
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
      {/* Terms and Conditions Modal */}
      <AnimatePresence>
        {isTermsOpen && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsTermsOpen(false)}
              className="absolute inset-0 bg-gray-900/60 backdrop-blur-sm"
            />
            
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-2xl bg-white rounded-3xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]"
            >
              <div className="flex items-center justify-between p-6 border-b border-gray-100 bg-gray-50/50">
                <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-rose-600" /> Terms of Service
                </h3>
                <button 
                  onClick={() => setIsTermsOpen(false)}
                  className="p-2 text-gray-400 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="p-6 overflow-y-auto custom-scrollbar flex-grow">
                <div className="prose prose-rose prose-sm sm:prose-base max-w-none text-gray-600">
                  <p className="font-semibold text-gray-900 mb-4">Last Updated: August 2026</p>
                  
                  <h4 className="text-gray-900 font-bold mb-2 text-lg">1. Medical Disclaimer</h4>
                  <p className="mb-4">DermaAI is an artificial intelligence-based educational tool designed to provide preliminary insights regarding skin conditions. It is <strong>NOT</strong> a diagnostic medical device and cannot replace professional medical consultation, diagnosis, or treatment.</p>
                  
                  <h4 className="text-gray-900 font-bold mb-2 text-lg">2. User Responsibility</h4>
                  <p className="mb-4">By using this application, you acknowledge that any information provided by DermaAI is for educational and informational purposes only. You agree to seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.</p>
                  
                  <h4 className="text-gray-900 font-bold mb-2 text-lg">3. Privacy and Data Security</h4>
                  <p className="mb-4">We are committed to protecting your privacy. All uploaded images are processed securely and are not stored permanently on our servers unless you explicitly opt-in for research purposes. Your personal health information is encrypted and handled in compliance with applicable healthcare data regulations.</p>
                  
                  <h4 className="text-gray-900 font-bold mb-2 text-lg">4. Limitation of Liability</h4>
                  <p className="mb-4">The creators, developers, and affiliated partners of DermaAI shall not be held liable for any direct, indirect, incidental, or consequential damages resulting from the use or inability to use the service, or from any actions taken based on the information provided by the application.</p>
                  
                  <h4 className="text-gray-900 font-bold mb-2 text-lg">5. Accuracy of AI Analysis</h4>
                  <p className="mb-4">While our machine learning models are highly advanced, artificial intelligence can make mistakes. The system may fail to identify serious conditions or may misidentify benign conditions as serious. A physical examination by a dermatologist is the only definitive way to diagnose skin conditions.</p>
                </div>
              </div>
              
              <div className="p-6 border-t border-gray-100 bg-gray-50/50 flex justify-end">
                <button 
                  onClick={() => setIsTermsOpen(false)}
                  className="px-6 py-2.5 bg-gray-900 text-white font-bold rounded-xl hover:bg-gray-800 transition-colors"
                >
                  I Understand
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
