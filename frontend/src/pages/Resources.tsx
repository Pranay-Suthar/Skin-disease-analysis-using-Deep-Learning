import React, { useEffect, useState } from 'react';
import Header from '../components/Header';
import YouTubePlayer from '../components/YouTubePlayer';
import { BookOpen, Video, ExternalLink, Loader, Search, HeartPulse, Clock, Sparkles, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { fetchSkinArticles, fetchYouTubeVideos, fetchHomeRemedies, Article, YouTubeVideo, HomeRemedy } from '../utils/resourcesApi';

const CONDITIONS = [
  'Actinic Keratosis',
  'Basal Cell Carcinoma',
  'Benign Keratosis',
  'Dermatofibroma',
  'Melanoma',
  'Melanocytic Nevus',
  'Squamous Cell Carcinoma',
  'Vascular Lesion'
];

export default function Resources() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [videos, setVideos] = useState<YouTubeVideo[]>([]);
  const [remedies, setRemedies] = useState<HomeRemedy[]>([]);
  
  const [loadingArticles, setLoadingArticles] = useState(true);
  const [loadingVideos, setLoadingVideos] = useState(true);
  const [loadingRemedies, setLoadingRemedies] = useState(true);

  // The condition filter
  const [selectedCondition, setSelectedCondition] = useState<string>('');

  useEffect(() => {
    const loadResources = async () => {
      try {
        setLoadingArticles(true);
        const fetchedArticles = await fetchSkinArticles(selectedCondition || 'skin dermatology care');
        setArticles(fetchedArticles);
      } catch (error) {
        console.error('Error loading articles:', error);
      } finally {
        setLoadingArticles(false);
      }

      try {
        setLoadingVideos(true);
        const fetchedVideos = await fetchYouTubeVideos(selectedCondition || 'dermatology skin care tips');
        setVideos(fetchedVideos);
      } catch (error) {
        console.error('Error loading videos:', error);
      } finally {
        setLoadingVideos(false);
      }

      try {
        setLoadingRemedies(true);
        // Default to Benign Keratosis remedies if no specific condition is selected
        const fetchedRemedies = await fetchHomeRemedies(selectedCondition || 'Benign Keratosis');
        setRemedies(fetchedRemedies);
      } catch (error) {
        console.error('Error loading remedies:', error);
      } finally {
        setLoadingRemedies(false);
      }
    };

    loadResources();
  }, [selectedCondition]);

  const handleSearchChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    setSelectedCondition(val);
    if (val) {
      const query = encodeURIComponent(`${val} skin condition treatment`);
      window.open(`https://www.youtube.com/results?search_query=${query}`, '_blank');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-slate-50 to-white pt-28 overflow-x-hidden">
      <Header />

      {/* Inline styles for marquee animation and custom scrollbar */}
      <style>{`
        @keyframes marquee {
          0% { transform: translateX(0); }
          100% { transform: translateX(calc(-50% - 12px)); }
        }
        .animate-marquee {
          animation: marquee 40s linear infinite;
        }
        .animate-marquee:hover {
          animation-play-state: paused;
        }
        
        .glass-card {
          background: rgba(255, 255, 255, 0.7);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.5);
        }
      `}</style>

      <main className="mx-auto max-w-7xl px-5 py-12">
        {/* Header */}
        <div className="mb-16 flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div className="animate-slide-up">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-50 text-rose-600 text-sm font-semibold mb-4">
              <Sparkles className="w-4 h-4" /> Discover & Learn
            </div>
            <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 mb-4 tracking-tight">
              Educational <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-600 to-indigo-500">Resources</span>
            </h1>
            <p className="text-lg text-slate-600 max-w-2xl">
              Curated medical literature, expert video tutorials, and daily care tips to help you understand your skin better.
            </p>
          </div>
          
          {/* Search Dropdown */}
          <div className="relative w-full md:w-80 animate-slide-left">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-rose-600/50" />
            </div>
            <select
              value={selectedCondition}
              onChange={handleSearchChange}
              className="block w-full pl-11 pr-10 py-4 text-base border-2 border-slate-200 focus:outline-none focus:ring-4 focus:ring-rose-500/20 focus:border-rose-500 sm:text-sm rounded-2xl bg-white/80 backdrop-blur-sm shadow-xl shadow-slate-200/50 cursor-pointer appearance-none text-slate-700 font-semibold transition-all hover:border-rose-300"
            >
              <option value="">Explore all conditions...</option>
              {CONDITIONS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-slate-400">
              <svg className="h-4 w-4 fill-current" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
              </svg>
            </div>
          </div>
        </div>

        {/* Videos Section */}
        <section className="mb-24">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-red-50 text-red-500 rounded-xl">
              <Video className="w-6 h-6" />
            </div>
            <h2 className="text-2xl font-bold text-slate-900">Video Tutorials</h2>
          </div>

          {loadingVideos ? (
            <div className="flex items-center justify-center py-12">
              <Loader className="w-8 h-8 text-rose-600 animate-spin mr-3" />
              <span className="text-slate-500 font-medium">Curating videos...</span>
            </div>
          ) : videos.length > 0 ? (
            <div className="w-screen relative overflow-hidden py-8 left-1/2 right-1/2 -ml-[50vw] -mr-[50vw]">
              <div className="absolute left-0 top-0 bottom-0 w-12 md:w-32 bg-gradient-to-r from-slate-50 to-transparent z-10 pointer-events-none"></div>
              <div className="absolute right-0 top-0 bottom-0 w-12 md:w-32 bg-gradient-to-l from-slate-50 to-transparent z-10 pointer-events-none"></div>
              
              <div className="flex gap-6 w-max animate-marquee">
                {[...videos, ...videos, ...videos, ...videos].map((video, idx) => (
                  <div key={`${video.id}-${idx}`} className="w-[420px] md:w-[520px] shrink-0 bg-white rounded-3xl p-3 shadow-xl shadow-slate-200/50 hover:shadow-2xl hover:shadow-rose-500/10 transition-all duration-300 border border-slate-100 group">
                    <div className="rounded-2xl overflow-hidden">
                      <YouTubePlayer
                        videoId={video.videoId}
                        title={video.title}
                        thumbnail={video.thumbnail}
                      />
                    </div>
                    <div className="p-4 mt-2">
                      <h3 className="font-bold text-lg text-slate-900 mb-2 line-clamp-2 group-hover:text-rose-600 transition-colors leading-snug">
                        {video.title}
                      </h3>
                      <p className="text-sm text-slate-500 mb-4 line-clamp-2 leading-relaxed">
                        {video.description}
                      </p>
                      <div className="flex items-center justify-between pt-4 border-t border-slate-100">
                        <span className="font-medium text-xs text-slate-400 bg-slate-50 px-2 py-1 rounded-md">{video.channel}</span>
                        <a
                          href={video.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1.5 text-red-600 hover:text-red-700 font-bold text-sm bg-red-50 hover:bg-red-100 px-3 py-1.5 rounded-full transition-colors"
                        >
                          Watch <ExternalLink className="w-3.5 h-3.5" />
                        </a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
             <div className="text-center py-16 bg-white rounded-3xl border border-dashed border-slate-300">
               <Video className="w-12 h-12 text-slate-300 mx-auto mb-3" />
               <p className="text-slate-500 font-medium">No videos found for this condition</p>
             </div>
          )}
        </section>

        {/* Premium Articles Section */}
        <section className="mb-24">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-amber-50 text-amber-600 rounded-xl">
              <BookOpen className="w-6 h-6" />
            </div>
            <h2 className="text-2xl font-bold text-slate-900">Featured Literature</h2>
          </div>

          {loadingArticles ? (
            <div className="flex items-center justify-center py-12">
              <Loader className="w-8 h-8 text-amber-600 animate-spin mr-3" />
              <span className="text-slate-500 font-medium">Fetching literature...</span>
            </div>
          ) : articles.length > 0 ? (
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {articles.map((article, idx) => (
                <a
                  key={`${article.id}-${idx}`}
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group flex flex-col bg-white rounded-3xl overflow-hidden shadow-lg shadow-slate-200/40 hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 border border-slate-100"
                >
                  {article.image && (
                    <div className="relative h-56 w-full overflow-hidden bg-slate-100">
                      <img
                        src={article.image}
                        alt={article.title}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700 ease-out"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-slate-900/80 via-slate-900/20 to-transparent opacity-80" />
                      
                      {/* Floating Category Badge */}
                      <div className="absolute top-4 left-4">
                        <span className="backdrop-blur-md bg-white/20 text-white border border-white/30 px-3 py-1.5 text-xs font-bold rounded-full uppercase tracking-wider">
                          {article.category}
                        </span>
                      </div>
                      
                      {/* Floating Source */}
                      <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center text-white">
                        <span className="font-semibold text-sm drop-shadow-md">{article.source}</span>
                        <span className="text-xs font-medium text-white/80">{article.publishedDate}</span>
                      </div>
                    </div>
                  )}
                  
                  <div className="p-6 flex flex-col flex-grow">
                    <h3 className="font-extrabold text-xl text-slate-900 mb-3 group-hover:text-amber-600 transition-colors leading-tight">
                      {article.title}
                    </h3>
                    <p className="text-slate-500 mb-6 line-clamp-3 leading-relaxed text-sm flex-grow">
                      {article.description}
                    </p>
                    <div className="flex items-center text-amber-600 font-bold text-sm mt-auto group-hover:gap-2 transition-all">
                      Read Full Article <ExternalLink className="w-4 h-4 ml-1 transition-transform group-hover:translate-x-1" />
                    </div>
                  </div>
                </a>
              ))}
            </div>
          ) : (
            <div className="text-center py-16 bg-white rounded-3xl border border-dashed border-slate-300">
              <BookOpen className="w-12 h-12 text-slate-300 mx-auto mb-3" />
              <p className="text-slate-500 font-medium">No articles available</p>
            </div>
          )}
        </section>

        {/* NEW: Home Remedies & Daily Care Section */}
        <section className="mb-20">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-indigo-50 text-indigo-600 rounded-xl">
              <HeartPulse className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-slate-900">Daily Care & Remedies</h2>
              {selectedCondition && (
                <p className="text-sm text-slate-500 mt-1">Specific advice for {selectedCondition}</p>
              )}
            </div>
          </div>

          {loadingRemedies ? (
             <div className="flex items-center justify-center py-12">
               <Loader className="w-8 h-8 text-indigo-600 animate-spin mr-3" />
               <span className="text-slate-500 font-medium">Loading remedies...</span>
             </div>
          ) : remedies.length > 0 ? (
            <div className="grid gap-6 lg:grid-cols-2">
              {remedies.map((remedy, idx) => (
                <div key={idx} className="bg-white rounded-3xl p-8 shadow-xl shadow-slate-200/40 border border-slate-100 hover:shadow-2xl transition-all duration-300 relative overflow-hidden group">
                  {/* Decorative background blob */}
                  <div className="absolute top-0 right-0 -mr-16 -mt-16 w-48 h-48 bg-indigo-50 rounded-full blur-3xl opacity-50 group-hover:opacity-80 transition-opacity pointer-events-none"></div>
                  
                  <h3 className="text-xl font-bold text-slate-900 mb-3 relative z-10 flex items-start gap-2">
                    {remedy.title.includes('⚠️') ? null : <Sparkles className="w-5 h-5 text-indigo-500 mt-0.5 shrink-0" />}
                    {remedy.title}
                  </h3>
                  
                  <p className="text-slate-600 mb-6 relative z-10">{remedy.description}</p>
                  
                  {remedy.ingredients && remedy.ingredients.length > 0 && (
                    <div className="mb-5 relative z-10">
                      <h4 className="text-sm font-bold text-slate-900 mb-2 uppercase tracking-wider text-indigo-700">What you need</h4>
                      <div className="flex flex-wrap gap-2">
                        {remedy.ingredients.map((ing, i) => (
                          <span key={i} className="px-3 py-1 bg-slate-100 text-slate-700 text-xs font-medium rounded-lg border border-slate-200">
                            {ing}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {remedy.steps && remedy.steps.length > 0 && (
                    <div className="mb-6 relative z-10">
                      <h4 className="text-sm font-bold text-slate-900 mb-3 uppercase tracking-wider text-indigo-700">Steps</h4>
                      <ul className="space-y-3">
                        {remedy.steps.map((step, i) => (
                          <li key={i} className="flex items-start gap-3 text-sm text-slate-600">
                            <CheckCircle2 className="w-5 h-5 text-indigo-500 shrink-0" />
                            <span className="pt-0.5">{step}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="flex items-center gap-4 border-t border-slate-100 pt-5 relative z-10">
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-700 bg-slate-50 px-3 py-1.5 rounded-lg">
                      <Clock className="w-4 h-4 text-indigo-600" />
                      {remedy.frequency}
                    </div>
                  </div>

                  {remedy.warning && (
                    <div className="mt-5 p-4 bg-red-50/80 backdrop-blur-sm rounded-xl border border-red-100 flex items-start gap-3 relative z-10">
                      <AlertTriangle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
                      <p className="text-sm text-red-800 font-medium leading-relaxed">{remedy.warning}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-16 bg-white rounded-3xl border border-dashed border-slate-300">
              <HeartPulse className="w-12 h-12 text-slate-300 mx-auto mb-3" />
              <p className="text-slate-500 font-medium">No home remedies found for this condition</p>
            </div>
          )}
        </section>

        {/* Medical Disclaimer */}
        <div className="mt-16 p-8 bg-slate-900 rounded-3xl relative overflow-hidden shadow-2xl">
          <div className="absolute top-0 right-0 w-64 h-64 bg-rose-500/10 rounded-full blur-3xl"></div>
          <div className="relative z-10">
            <h3 className="text-xl font-bold text-white mb-3 flex items-center gap-2">
              <AlertTriangle className="w-6 h-6 text-yellow-400" /> 
              Medical Disclaimer
            </h3>
            <p className="text-slate-300 text-sm leading-relaxed max-w-4xl">
              All resources provided on this page are for educational purposes only. They are not a substitute for professional medical advice, diagnosis, or treatment. 
              Always consult with a qualified dermatologist or healthcare provider for medical concerns specific to your skin condition. Never disregard professional medical advice or delay in seeking it because of something you have read on this website.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
