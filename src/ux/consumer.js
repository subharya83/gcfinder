import React, { useState } from 'react';
import { 
  ArrowLeft, 
  User, 
  Award, 
  Coins, 
  Upload, 
  Clock, 
  AlertTriangle,
  Home,
  Gift,
  Share2
} from 'lucide-react';

const ConsumerDashboard = () => {
  const [activeTab, setActiveTab] = useState('home');
  
  // Mock data for contributions
  const contributions = [
    {
      id: 1,
      type: 'photo',
      description: 'Menu photo for Joe\'s Café - Validated',
      status: 'validated',
      reward: '+5 Tokens',
      thumbnail: '🍕'
    },
    {
      id: 2,
      type: 'review',
      description: 'Review for Bella\'s Bistro - Pending',
      status: 'pending',
      reward: 'Pending',
      thumbnail: '⭐'
    },
    {
      id: 3,
      type: 'photo',
      description: 'Hours update for Corner Deli - Validated',
      status: 'validated',
      reward: '+3 Tokens',
      thumbnail: '🕒'
    },
    {
      id: 4,
      type: 'review',
      description: 'Fake review submission - Rejected',
      status: 'rejected',
      reward: '-0.1 Reputation',
      thumbnail: '❌',
      penalty: true
    },
    {
      id: 5,
      type: 'photo',
      description: 'Menu photo for Pizza Palace - Validated',
      status: 'validated',
      reward: '+4 Tokens',
      thumbnail: '🍕'
    }
  ];

  const reputationScore = 85;
  const totalTokens = 120;
  const totalContributions = 15;
  const maxReputation = 100;

  const getStatusIcon = (status) => {
    switch (status) {
      case 'validated':
        return <Coins className="w-4 h-4 text-blue-600" />;
      case 'pending':
        return <Clock className="w-4 h-4 text-gray-500" />;
      case 'rejected':
        return <AlertTriangle className="w-4 h-4 text-red-600" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status, penalty = false) => {
    if (penalty) return 'text-red-600';
    switch (status) {
      case 'validated':
        return 'text-blue-600';
      case 'pending':
        return 'text-gray-500';
      case 'rejected':
        return 'text-red-600';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <div className="bg-white min-h-screen max-w-sm mx-auto border border-gray-200">
      {/* Header - 10% */}
      <div className="h-16 bg-white border-b border-gray-100 flex items-center justify-between px-4">
        <ArrowLeft className="w-6 h-6 text-black" />
        <h1 className="text-xl font-bold text-black">Your Contributions</h1>
        <User className="w-6 h-6 text-black" />
      </div>

      {/* Main Section - 70% */}
      <div className="px-4 py-4 pb-24">
        {/* Reputation Score Card */}
        <div className="bg-white border border-gray-300 rounded-lg p-4 mb-4">
          <div className="flex flex-col items-center">
            {/* Circular Progress Indicator */}
            <div className="relative w-20 h-20 mb-2">
              <svg className="w-20 h-20 transform -rotate-90" viewBox="0 0 80 80">
                <circle
                  cx="40"
                  cy="40"
                  r="36"
                  stroke="rgb(229, 231, 235)"
                  strokeWidth="8"
                  fill="none"
                />
                <circle
                  cx="40"
                  cy="40"
                  r="36"
                  stroke="rgb(37, 99, 235)"
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={`${(reputationScore / maxReputation) * 226.2} 226.2`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-lg font-bold text-black">{reputationScore}</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Award className="w-5 h-5 text-blue-600" />
              <span className="text-base text-black">{reputationScore}/100 - Trusted Contributor</span>
            </div>
          </div>
        </div>

        {/* Recent Contributions List */}
        <div className="space-y-3 mb-4 max-h-80 overflow-y-auto">
          {contributions.map((contribution) => (
            <div key={contribution.id} className="bg-white border border-gray-300 rounded-lg p-3">
              <div className="flex items-center gap-3">
                {/* Thumbnail */}
                <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center text-lg">
                  {contribution.thumbnail}
                </div>
                
                {/* Content */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-black truncate">{contribution.description}</p>
                  {contribution.penalty && (
                    <div className="flex items-center gap-1 mt-1">
                      <AlertTriangle className="w-3 h-3 text-red-600" />
                      <span className="text-xs text-red-600">Invalid Submission</span>
                    </div>
                  )}
                </div>
                
                {/* Reward Status */}
                <div className="flex items-center gap-1">
                  {getStatusIcon(contribution.status)}
                  <span className={`text-sm font-medium ${getStatusColor(contribution.status, contribution.penalty)}`}>
                    {contribution.reward}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Summary Metrics */}
        <div className="flex gap-3 mb-6">
          <div className="flex-1 bg-white border border-gray-300 rounded-lg p-3">
            <div className="flex items-center gap-2">
              <Coins className="w-5 h-5 text-blue-600" />
              <div>
                <p className="text-xs text-gray-600">Total Tokens</p>
                <p className="text-lg font-bold text-blue-600">{totalTokens}</p>
              </div>
            </div>
          </div>
          <div className="flex-1 bg-white border border-gray-300 rounded-lg p-3">
            <div className="flex items-center gap-2">
              <Upload className="w-5 h-5 text-black" />
              <div>
                <p className="text-xs text-gray-600">Contributions</p>
                <p className="text-lg font-bold text-black">{totalContributions}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer - 20% */}
      <div className="fixed bottom-0 left-1/2 transform -translate-x-1/2 w-full max-w-sm bg-white border-t border-gray-200">
        {/* CTA Button */}
        <div className="px-4 py-3">
          <button className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium text-base hover:bg-blue-700 transition-colors">
            Submit New Data
          </button>
        </div>
        
        {/* Bottom Navigation */}
        <div className="flex items-center justify-around py-2 border-t border-gray-100">
          <button 
            onClick={() => setActiveTab('home')}
            className={`flex flex-col items-center gap-1 py-2 px-3 ${activeTab === 'home' ? 'text-blue-600' : 'text-gray-400'}`}
          >
            <Home className="w-5 h-5" />
            <span className="text-xs">Home</span>
          </button>
          <button 
            onClick={() => setActiveTab('rewards')}
            className={`flex flex-col items-center gap-1 py-2 px-3 ${activeTab === 'rewards' ? 'text-blue-600' : 'text-gray-400'}`}
          >
            <Gift className="w-5 h-5" />
            <span className="text-xs">Rewards</span>
          </button>
          <button 
            onClick={() => setActiveTab('profile')}
            className={`flex flex-col items-center gap-1 py-2 px-3 ${activeTab === 'profile' ? 'text-blue-600' : 'text-gray-400'}`}
          >
            <User className="w-5 h-5" />
            <span className="text-xs">Profile</span>
          </button>
          <button 
            onClick={() => setActiveTab('sharing')}
            className="flex flex-col items-center gap-1 py-2 px-3 text-gray-400"
          >
            <Share2 className="w-5 h-5" />
            <span className="text-xs">Data Sharing</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConsumerDashboard;
