import React, { useState } from 'react';
import { 
  ArrowLeft, 
  Settings, 
  BarChart3, 
  TrendingUp,
  AlertTriangle,
  Send,
  CheckCircle,
  XCircle,
  Home,
  Gift,
  User,
  Share2,
  Clock
} from 'lucide-react';

const VendorDashboard = () => {
  const [activeTab, setActiveTab] = useState('rewards');
  const [rewardType, setRewardType] = useState('');
  const [expirationTime, setExpirationTime] = useState('');

  // Mock data for contributions feedback
  const contributionFeedback = [
    {
      id: 1,
      type: 'validated',
      description: 'Menu photo validated: +5 tokens to user',
      user: 'Sarah M.',
      timestamp: '2 hrs ago'
    },
    {
      id: 2,
      type: 'validated',
      description: 'Hours update confirmed: +3 tokens to user',
      user: 'Mike R.',
      timestamp: '3 hrs ago'
    },
    {
      id: 3,
      type: 'rejected',
      description: 'Incorrect stock report: -0.1 reputation to user',
      user: 'Alex K.',
      timestamp: '4 hrs ago'
    },
    {
      id: 4,
      type: 'validated',
      description: 'Customer review verified: +4 tokens to user',
      user: 'Emma L.',
      timestamp: '5 hrs ago'
    },
    {
      id: 5,
      type: 'validated',
      description: 'Price update confirmed: +2 tokens to user',
      user: 'David P.',
      timestamp: '6 hrs ago'
    }
  ];

  const rewardOptions = [
    '10% Discount',
    '15% Discount',
    '20% Discount',
    'Free Coffee',
    'Free Appetizer',
    'Buy 1 Get 1 Free'
  ];

  const handleBroadcast = () => {
    if (rewardType && expirationTime) {
      // Handle broadcast logic
      alert(`Broadcast sent: ${rewardType}, expires in ${expirationTime}`);
      setRewardType('');
      setExpirationTime('');
    }
  };

  const getContributionIcon = (type) => {
    return type === 'validated' ? (
      <CheckCircle className="w-4 h-4 text-green-600" />
    ) : (
      <XCircle className="w-4 h-4 text-red-600" />
    );
  };

  return (
    <div className="bg-white min-h-screen max-w-sm mx-auto border border-gray-200">
      {/* Header - 10% */}
      <div className="h-16 bg-white border-b border-gray-100 flex items-center justify-between px-4">
        <ArrowLeft className="w-6 h-6 text-black" />
        <h1 className="text-xl font-bold text-black">Business Dashboard</h1>
        <Settings className="w-6 h-6 text-black" />
      </div>

      {/* Main Section - 70% */}
      <div className="px-4 py-4 pb-24">
        {/* Data Insights Card */}
        <div className="bg-white border border-gray-300 rounded-lg p-4 mb-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-base font-bold text-black">Real-Time Insights</h2>
            <BarChart3 className="w-5 h-5 text-blue-600" />
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-green-600" />
              <span className="text-sm text-black">Current Demand: High (20 pre-orders)</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-blue-600" />
              <span className="text-sm text-black">Popular Item: Margherita Pizza (15 orders)</span>
            </div>
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-red-600" />
              <span className="text-sm text-red-600">Stock Alert: Low on tomatoes (10% remaining)</span>
            </div>
          </div>
        </div>

        {/* Incentive Broadcast Section */}
        <div className="bg-white border border-gray-300 rounded-lg p-4 mb-4">
          <h2 className="text-base font-bold text-black mb-2">Broadcast Incentive</h2>
          <p className="text-sm text-gray-600 mb-4">Offer a deal to validators for data updates</p>
          
          <div className="space-y-3 mb-4">
            {/* Reward Type Dropdown */}
            <div>
              <label className="block text-xs text-gray-600 mb-1">Reward Type</label>
              <select 
                value={rewardType}
                onChange={(e) => setRewardType(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-lg text-sm bg-white"
              >
                <option value="">Select reward type</option>
                {rewardOptions.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>
            
            {/* Expiration Time Input */}
            <div>
              <label className="block text-xs text-gray-600 mb-1">Deal Expires In</label>
              <input
                type="text"
                value={expirationTime}
                onChange={(e) => setExpirationTime(e.target.value)}
                placeholder="e.g., 4 hrs"
                className="w-full p-2 border border-gray-300 rounded-lg text-sm"
              />
            </div>
            
            {/* Broadcast Button */}
            <button 
              onClick={handleBroadcast}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
            >
              <Send className="w-4 h-4" />
              Broadcast to Validators
            </button>
          </div>
          
          {/* Last Broadcast Status */}
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <Clock className="w-4 h-4" />
            <span>Last Broadcast: 50% off coffee, expires in 2 hrs</span>
          </div>
        </div>

        {/* Contribution Feedback */}
        <div className="bg-white border border-gray-300 rounded-lg p-4">
          <h2 className="text-base font-bold text-black mb-3">Contribution Feedback</h2>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {contributionFeedback.map((feedback) => (
              <div key={feedback.id} className="flex items-start gap-3 p-2 bg-gray-50 rounded-lg">
                <div className="mt-1">
                  {getContributionIcon(feedback.type)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-black">{feedback.description}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs text-gray-500">{feedback.user}</span>
                    <span className="text-xs text-gray-400">•</span>
                    <span className="text-xs text-gray-500">{feedback.timestamp}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer - 20% */}
      <div className="fixed bottom-0 left-1/2 transform -translate-x-1/2 w-full max-w-sm bg-white border-t border-gray-200">
        {/* CTA Button */}
        <div className="px-4 py-3">
          <button className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium text-base hover:bg-blue-700 transition-colors flex items-center justify-center gap-2">
            <BarChart3 className="w-5 h-5" />
            View Full Analytics
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

export default VendorDashboard;
