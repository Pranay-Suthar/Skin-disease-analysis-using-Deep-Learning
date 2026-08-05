import React, { useEffect, useState } from 'react';
import Header from '../components/Header';
import YouTubePlayer from '../components/YouTubePlayer';
import { BookOpen, Video, ExternalLink, Loader } from 'lucide-react';
import { fetchSkinArticles, fetchYouTubeVideos, Article, YouTubeVideo } from '../utils/resourcesApi';

export default function Resources() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [videos, setVideos] = useState<YouTubeVideo[]>([]);
  const [loadingArticles, setLoadingArticles] = useState(true);
  const [loadingVideos, setLoadingVideos] = useState(true);

  useEffect(() => {
    const loadResources = async () => {
      try {
        setLoadingArticles(true);
        const fetchedArticles = await fetchSkinArticles('skin dermatology care');
        setArticles(fetchedArticles);
      } catch (error) {
        console.error('Error loading articles:', error);
      } finally {
        setLoadingArticles(false);
      }

      try {
        setLoadingVideos(true);
        const fetchedVideos = await fetchYouTubeVideos('dermatology skin care tips');
        setVideos(fetchedVideos);
      } catch (error) {
        console.error('Error loading videos:', error);
      } finally {
        setLoadingVideos(false);
      }
    };

    loadResources();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-blue-50 to-white pt-28">
      <Header />

      <main className="mx-auto max-w-7xl px-5 py-12">
        {/* Header */}
        <div className="mb-16">
          <h1 className="page-title mb-3 flex items-center gap-3">
            <BookOpen className="w-10 h-10 text-teal-600" />
            Educational Resources
          </h1>
          <p className="page-subtitle">Learn about skin health from trusted sources</p>
        </div>

        {/* Videos Section */}
        <section className="mb-20">
          <div className="flex items-center gap-3 mb-8">
            <Video className="w-7 h-7 text-teal-600" />
            <h2 className="text-3xl font-bold text-gray-900">Video Tutorials</h2>
          </div>

          {loadingVideos ? (
            <div className="flex items-center justify-center py-12">
              <Loader className="w-8 h-8 text-teal-600 animate-spin mr-3" />
              <span className="text-gray-600">Loading videos...</span>
            </div>
          ) : videos.length > 0 ? (
            <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
              {videos.map((video) => (
                <div key={video.id} className="card group hover:shadow-xl transition-all duration-300">
                  <YouTubePlayer
                    videoId={video.videoId}
                    title={video.title}
                    thumbnail={video.thumbnail}
                  />
                  <div className="mt-4">
                    <h3 className="font-bold text-lg text-gray-900 mb-2 line-clamp-2 group-hover:text-teal-600 transition-colors">
                      {video.title}
                    </h3>
                    <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                      {video.description}
                    </p>
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span className="font-semibold text-gray-700">{video.channel}</span>
                      <span>{video.publishedAt}</span>
                    </div>
                    <a
                      href={video.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-4 inline-flex items-center gap-2 text-teal-600 hover:text-teal-700 font-semibold text-sm"
                    >
                      Watch on YouTube
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 bg-gray-50 rounded-lg">
              <Video className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-600">No videos available at the moment</p>
            </div>
          )}
        </section>

        {/* Articles Section */}
        <section>
          <div className="flex items-center gap-3 mb-8">
            <BookOpen className="w-7 h-7 text-teal-600" />
            <h2 className="text-3xl font-bold text-gray-900">Featured Articles</h2>
          </div>

          {loadingArticles ? (
            <div className="flex items-center justify-center py-12">
              <Loader className="w-8 h-8 text-teal-600 animate-spin mr-3" />
              <span className="text-gray-600">Loading articles...</span>
            </div>
          ) : articles.length > 0 ? (
            <div className="grid gap-8 md:grid-cols-2">
              {articles.map((article) => (
                <a
                  key={article.id}
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="card group hover:shadow-xl transition-all duration-300 overflow-hidden cursor-pointer"
                >
                  {article.image && (
                    <div className="relative h-48 mb-4 rounded-lg overflow-hidden bg-gray-200">
                      <img
                        src={article.image}
                        alt={article.title}
                        className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                  )}
                  
                  <div className="flex flex-col h-full">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="inline-block px-3 py-1 bg-teal-100 text-teal-700 text-xs font-semibold rounded-full">
                        {article.category}
                      </span>
                      <span className="text-xs text-gray-500">{article.publishedDate}</span>
                    </div>

                    <h3 className="font-bold text-lg text-gray-900 mb-2 line-clamp-2 group-hover:text-teal-600 transition-colors flex-grow">
                      {article.title}
                    </h3>

                    <p className="text-sm text-gray-600 mb-4 line-clamp-3">
                      {article.description}
                    </p>

                    <div className="flex items-center justify-between text-sm">
                      <span className="font-semibold text-gray-700">{article.source}</span>
                      <ExternalLink className="w-4 h-4 text-teal-600 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </div>
                </a>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 bg-gray-50 rounded-lg">
              <BookOpen className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-600">No articles available at the moment</p>
            </div>
          )}
        </section>

        {/* Disclaimer */}
        <div className="mt-16 p-6 bg-blue-50 border-l-4 border-blue-600 rounded-lg">
          <h3 className="font-bold text-blue-900 mb-2">📚 Educational Information</h3>
          <p className="text-blue-800 text-sm">
            All resources provided on this page are for educational purposes only. They are not a substitute for professional medical advice, diagnosis, or treatment. 
            Always consult with a qualified dermatologist or healthcare provider for medical concerns specific to your skin condition.
          </p>
        </div>
      </main>
    </div>
  );
}
