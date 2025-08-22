import React, { useState } from 'react';
import { 
  ArrowLeft, 
  HelpCircle, 
  Check,
  X,
  Coins,
  AlertTriangle,
  Trophy,
  Camera,
  MessageSquare,
  Clock,
  TrendingUp,
  Home,
  Gift,
  User,
  Share2
} from 'lucide-react';

const ValidatorDashboard = () => {
  const [activeTab, setActiveTab] = useState('rewards');
  const [stakeAmount, setStakeAmount] = useState('10');
  const [currentSubmission, setCurrentSubmission] = useState(0);

  // Mock data for pending submissions
  const pendingSubmissions = [
    {
      id: 1,
      type: 'photo',
      title: 'Menu Photo',
      content: 'New lunch menu for Joe\'s Café',
      thumbnail: '🍕',
      submittedBy: 'User123',
      timeAgo: '1 hr ago',
      preview: 'Menu photo showing daily specials and prices'
    },
    {
      id: 2,
      type: 'review',
      title: 'Customer Review',
      content: 'Great service at Bella\'s Bistro',
      thumbnail: '⭐',
      submittedBy: 'FoodLover88',
      timeAgo: '2 hrs ago',
      preview: 'Review: "Excellent pasta and friendly staff. Highly recommend!"'
    },
    {
      id: 3,
      type: 'hours',
      title: 'Hours Update',
      content: 'Updated hours for Corner Deli',
      thumbnail: '🕒',
      submittedBy: 'LocalGuide',
      timeAgo: '3 hrs ago',
      preview: 'New hours: Mon-Fri 7AM-9PM, Sat-Sun 8AM-8PM'
    }
  ];

  // Mock data for validation history
  const validationHistory = [
    {
      id: 1,
      type: 'Menu Photo',
      outcome: 'Approved: +0.02 Reputation',
      reward: '+2 Tokens',
      status: 'success',
      timeAgo: '2 hrs ago'
    },
    {
      id: 2,
      type: 'Stock Update',
      outcome: 'Approved: +0.02 Reputation',
      reward: '+3 Tokens',
      status: 'success',
      timeAgo: '4 hrs ago'
    },
    {
      id: 3,
      type: 'Customer Review',
      outcome: 'Against Consensus: -0.05 Reputation',
      reward: 'Stake Penalty: -5 Tokens',
      status: 'penalty',
      timeAgo: '1 day ago'
    },
    {
      id: 4,
      type: 'Hours Update',
      outcome: 'Approved: +0.02 Reputation',
      reward: '+1 Token',
      status: 'success',
      timeAgo: '2 days ago'
    },
    {
      id: 5,
      type: 'Menu Photo',
      outcome: 'Approved: +0.02 Reputation',
      reward: '+2 Tokens',
      status: 'success',
      timeAgo: '3 days ago'
    }
  ];

  const handleVote = (decision) => {
    if (!stakeAmount || parseFloat(stakeAmount) <= 0) {
      alert('Please enter a valid stake amount');
      return;
    }
    
    alert(`Vote submitted: ${decision} with ${stakeAmount} tokens staked`);
    
    // Move to next submission or show completion
    if (currentSubmission < pendingSubmissions.length - 1) {
      setCurrentSubmission(currentSubmission + 1);
    } else {
      alert('No more submissions to validate');
    }
    setStakeAmount('10'); // Reset stake amount
  };

  const getSubmissionIcon = (type) => {
    switch (type) {
      case 'photo':
        return <Camera className="w-5 h-5 text-blue-600" />;
      case 'review':
        return <MessageSquare className="w-5 h-5 text-blue-600" />;
      case 'hours':
        return <Clock className="w-5 h-5 text-blue-600" />;
      default:
        return <Camera className="w-5 h-5 text-blue-600" />;
    }
  };

  const getOutcomeColor = (status) => {
    return status === 'success' ? 'text-green-600' : 'text-red-600';
  };

  const getRewardIcon = (status) => {
    return status === 'success' ? (
      <Coins className="w-4 h-4 text-blue-600" />
    ) : (
      <AlertTriangle className="w-4 h-4 text-red-600" />
    );
  };

  const currentSub = pendingSubmissions[currentSubmission];

  return (
    <div className="bg-white min-h-screen max-w-sm mx-auto border border-gray-200">
      {/* Header - 10% */}
      <div className="h-16 bg-white border-b border-gray-100 flex items-center justify-between px-4">
        <ArrowLeft className="w-6 h-6 text-black" />
        <h1 className="text-xl font-bold text-black">Validator Hub</h1>
        <HelpCircle className="w-6 h-6 text-black" />
      </div>

      {/* Main Section - 70% */}
      <div className="px-4 py-4 pb-24">
        {/* Pending Validation Card */}
        {currentSub && (
          <div className="bg-white border border-gray-300 rounded-lg p-4 mb-4">
            <div className="flex items-center gap-2 mb-3">
              {getSubmissionIcon(currentSub.type)}
              <h2 className="text-base font-bold text-black">New Submission</h2>
            </div>
            
            {/* Content Preview */}
            <div className="bg-gray-50 rounded-lg p-3 mb-3">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center text-lg">
                  {currentSub.thumbnail}
                </div>
                <div className="flex-1">
                  <h3 className="font-medium text-black">{currentSub.title}</h3>
                  <p className="text-sm text-gray-600">{currentSub.content}</p>
                </div>
              </div>
              <p className="text-sm text-gray-700">{currentSub.preview}</p>
            </div>
            
            {/* Metadata */}
            <p className="text-sm text-gray-600 mb-4">
              Submitted {currentSub.timeAgo} by {currentSub.submittedBy}
            </p>
            
            {/* Stake Input */}
            <div className="mb-4">
              <label className="block text-sm text-gray-600 mb-1">Stake Amount (Tokens)</label>
              <input
                type="number"
                value={stakeAmount}
                onChange={(e) => setStakeAmount(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-lg text-sm"
                min="1"
                step="1"
              />
            </div>
            
            {/* Voting Buttons */}
            <div className="flex gap-3">
              <button 
                onClick={() => handleVote('Approve')}
                className="flex-1 flex items-center justify-center gap-2 py-2 px-4 border-2 border-green-600 text-green-600 rounded-lg hover:bg-green-50 transition-colors"
              >
                <Check className="w-4 h-4" />
                <span className="text-sm font-medium">Approve</span>
              </button>
              <button 
                onClick={() => handleVote('Reject')}
                className="flex-1 flex items-center justify-center gap-2 py-2 px-4 border-2 border-red-600 text-red-600 rounded-lg hover:bg-red-50 transition-colors"
              >
                <X className="w-4 h-4" />
                <span className="text-sm font-medium">Reject</span>
              </button>
            </div>
          </div>
        )}

        {/* Validation History */}
        <div className="bg-white border border-gray-300 rounded-lg p-4 mb-4">
          <h2 className="text-base font-bold text-black mb-3">Validation History</h2>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {validationHistory.map((validation) => (
              <div key={validation.id} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                <div className="flex-1">
                  <p className="text-sm font-medium text-black">{validation.type}</p>
                  <p className={`text-xs ${getOutcomeColor(validation.status)}`}>
                    {validation.outcome}
                  </p>
                  <p className="text-xs text-gray-500">{validation.timeAgo}</p>
                </div>
                <div className="flex items-center gap-1">
                  {getRewardIcon(validation.status)}
                  <span className={`text-sm font-medium ${getOutcomeColor(validation.status)}`}>
                    {validation.reward}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Validator Stats */}
        <div className="bg-white border border-gray-300 rounded-lg p-4">
          <h2 className="text-base font-bold text-black mb-3">Validator Stats</h2>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-green-600" />
                <span className="text-sm text-black">Validation Accuracy</span>
              </div>
              <span className="text-sm font-bold text-black">92%</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Coins className="w-4 h-4 text-blue-600" />
                <span className="text-sm text-black">Total Rewards</span>
              </div>
              <span className="text-sm font-bold text-blue-600">50 Tokens</span>
            </div>
          </div>
        </div>
      </div>

      {/* Footer - 20% */}
      <div className="fixed bottom-0 left-1/2 transform -translate-x-1/2 w-full max-w-sm bg-white border-t border-gray-200">
        {/* CTA Button */}
        <div className="px-4 py-3">
          <button className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium text-base hover:bg-blue-700 transition-colors flex items-center justify-center gap-2">
            <Trophy className="w-5 h-5" />
            View Leaderboard
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

export default ValidatorDashboard;
