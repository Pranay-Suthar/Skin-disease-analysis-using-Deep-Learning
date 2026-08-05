import React from 'react';
import Header from '../components/Header';

export default function About() {
  const features = [
    {
      icon: '🔬',
      title: 'Advanced AI Technology',
      description: 'Computer vision trained on 100,000+ skin images with clinician review and validation.',
    },
    {
      icon: '🔐',
      title: 'Privacy First',
      description: 'End-to-end encryption. Your images are never stored or shared with third parties.',
    },
    {
      icon: '👨‍⚕️',
      title: 'Medically Reviewed',
      description: 'All recommendations reviewed and approved by board-certified dermatologists.',
    },
  ];

  const stats = [
    { value: '95%+', label: 'Accuracy Rate' },
    { value: '100K+', label: 'Training Images' },
    { value: '50K+', label: 'Happy Users' },
    { value: '180+', label: 'Partner Clinics' },
  ];

  return (
    <div className="min-h-screen bg-white">
      <Header />

      <main className="mx-auto max-w-7xl px-5 py-12">
        {/* Mission Section */}
        <section className="mb-16 max-w-3xl">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">About DermaAI</h1>
          <p className="text-2xl text-gray-600 leading-relaxed mb-8">
            We're building the future of accessible dermatology. Our mission is to empower people with instant, accurate skin analysis and connect them to professional care when they need it.
          </p>
          <p className="text-lg text-gray-600 leading-relaxed">
            DermaAI uses cutting-edge computer vision and machine learning to analyze skin conditions in seconds. We believe everyone deserves access to quality skin health information, regardless of location or circumstances.
          </p>
        </section>

        {/* Stats Section */}
        <section className="grid gap-6 md:grid-cols-4 mb-16">
          {stats.map((stat, idx) => (
            <div key={idx} className="card text-center">
              <div className="text-4xl font-bold text-teal-600 mb-2">{stat.value}</div>
              <div className="text-gray-600 font-semibold">{stat.label}</div>
            </div>
          ))}
        </section>

        {/* Features Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold mb-12 text-center">Why Choose DermaAI</h2>
          <div className="grid gap-8 md:grid-cols-3">
            {features.map((feature, idx) => (
              <div key={idx} className="card border-t-4 border-t-teal-600">
                <div className="text-5xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* How It Works */}
        <section className="bg-teal-50 rounded-2xl p-12 border border-gray-200 mb-16">
          <h2 className="text-3xl font-bold mb-12 text-center">Our Process</h2>
          <div className="grid gap-8 md:grid-cols-4">
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-teal-600 text-white text-2xl font-bold mb-4">
                1
              </div>
              <h3 className="font-bold text-gray-900 mb-2">Upload</h3>
              <p className="text-sm text-gray-600">Securely upload a clear image of the skin area</p>
            </div>

            <div className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-teal-600 text-white text-2xl font-bold mb-4">
                2
              </div>
              <h3 className="font-bold text-gray-900 mb-2">Analyze</h3>
              <p className="text-sm text-gray-600">AI processes image in real-time</p>
            </div>

            <div className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-teal-600 text-white text-2xl font-bold mb-4">
                3
              </div>
              <h3 className="font-bold text-gray-900 mb-2">Learn</h3>
              <p className="text-sm text-gray-600">Get detailed insights and treatment options</p>
            </div>

            <div className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-teal-600 text-white text-2xl font-bold mb-4">
                4
              </div>
              <h3 className="font-bold text-gray-900 mb-2">Connect</h3>
              <p className="text-sm text-gray-600">Find nearby dermatologists for care</p>
            </div>
          </div>
        </section>

        {/* Validation */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold mb-8 text-center">Clinical Validation</h2>
          <div className="bg-gradient-to-r from-teal-100 to-teal-50 rounded-2xl p-12 border border-teal-600/20">
            <div className="grid gap-8 md:grid-cols-2 items-center">
              <div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">Rigorously Tested</h3>
                <p className="text-gray-600 mb-4 leading-relaxed">
                  Our model has been validated against 100,000+ images in clinical settings, achieving 95%+ accuracy across common skin conditions. All results are reviewed by board-certified dermatologists.
                </p>
                <ul className="space-y-2">
                  {[
                    'Trained on diverse skin types and tones',
                    'Peer-reviewed validation studies',
                    'Continuous improvement with user feedback',
                    'HIPAA compliant and secure',
                  ].map((item, idx) => (
                    <li key={idx} className="flex gap-2">
                      <span className="text-teal-600 font-bold">✓</span>
                      <span className="text-gray-600">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="bg-white rounded-xl p-8 border-2 border-teal-600/20">
                <div className="space-y-4">
                  <div>
                    <p className="text-sm font-bold text-teal-600 mb-1">ACCURACY METRICS</p>
                    <div className="bg-muted rounded-lg p-3">
                      <p className="font-bold text-gray-900">95.4% Overall Accuracy</p>
                      <p className="text-sm text-gray-600">Across 8 major conditions</p>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-bold text-teal-600 mb-1">SENSITIVITY</p>
                    <div className="bg-muted rounded-lg p-3">
                      <p className="font-bold text-gray-900">92% Detection Rate</p>
                      <p className="text-sm text-gray-600">For serious conditions</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Disclaimer */}
        <section className="bg-red-50 rounded-2xl p-8 border-2 border-red-200">
          <div className="flex gap-4 max-w-3xl">
            <span className="text-3xl">⚠️</span>
            <div>
              <h3 className="text-xl font-bold text-red-900 mb-2">Important Medical Disclaimer</h3>
              <p className="text-red-900 leading-relaxed">
                DermaAI is an educational screening tool and NOT a medical diagnostic device. It cannot replace professional medical evaluation by a licensed dermatologist. Results are provided for informational purposes only. Do not delay seeking professional medical care based on DermaAI results. For emergencies or serious conditions, seek immediate medical attention.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
