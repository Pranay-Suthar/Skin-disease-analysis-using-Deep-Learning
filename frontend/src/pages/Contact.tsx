import React, { useState } from 'react';
import Header from '../components/Header';

export default function Contact() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: '',
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    alert('Thank you for your message! We will respond soon.');
    setFormData({ name: '', email: '', message: '' });
  };

  return (
    <div className="min-h-screen bg-white">
      <Header />

      <main className="mx-auto max-w-7xl px-5 py-12">
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-3">Contact Us</h1>
          <p className="text-lg text-gray-600">Get in touch with our support team</p>
        </div>

        <div className="grid gap-12 lg:grid-cols-2">
          {/* Form */}
          <div className="card-elevated">
            <h2 className="text-2xl font-bold mb-6 text-gray-900">Send a Message</h2>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-bold text-gray-900 mb-2">Name</label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  placeholder="Your name"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-600"
                />
              </div>

              <div>
                <label className="block text-sm font-bold text-gray-900 mb-2">Email</label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  placeholder="you@example.com"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-600"
                />
              </div>

              <div>
                <label className="block text-sm font-bold text-gray-900 mb-2">Message</label>
                <textarea
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  required
                  placeholder="Your message..."
                  rows={5}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-600 resize-none"
                />
              </div>

              <button
                type="submit"
                className="w-full px-6 py-3 bg-teal-600 text-white font-bold rounded-lg hover:bg-teal-700 transition"
              >
                Send Message
              </button>
            </form>
          </div>

          {/* Contact Info */}
          <div className="space-y-6">
            <div className="card">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Direct Contact</h3>
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-bold text-teal-600 mb-1">Email</p>
                  <p className="font-bold text-gray-900">support@dermaai.health</p>
                </div>
                <div>
                  <p className="text-sm font-bold text-teal-600 mb-1">Phone</p>
                  <p className="font-bold text-gray-900">(800) 555-0126</p>
                </div>
                <div>
                  <p className="text-sm font-bold text-teal-600 mb-1">Hours</p>
                  <p className="font-bold text-gray-900">Mon-Fri, 8 AM-6 PM PT</p>
                </div>
              </div>
            </div>

            <div className="card bg-teal-50 border-teal-600 border-2">
              <h3 className="font-bold text-gray-900 mb-2">💡 Need Help?</h3>
              <p className="text-sm text-gray-600 mb-4">
                Our support team is here to help you get the most out of DermaAI.
              </p>
              <a href="/hospitals" className="text-teal-600 font-bold hover:text-teal-700">
                Find Care →
              </a>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
