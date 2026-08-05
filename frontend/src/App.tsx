import React from 'react';
import { BrowserRouter, Routes, Route, Link, useNavigate } from 'react-router-dom';
import { Icon } from '@iconify/react';

// Pages
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import Hospitals from './pages/Hospitals';
import Resources from './pages/Resources';
import About from './pages/About';
import Contact from './pages/Contact';
import SkinBot from './pages/SkinBot';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/analyze" element={<Analyze />} />
        <Route path="/hospitals" element={<Hospitals />} />
        <Route path="/resources" element={<Resources />} />
        <Route path="/about" element={<About />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/skinbot" element={<SkinBot />} />
      </Routes>
    </BrowserRouter>
  );
}
