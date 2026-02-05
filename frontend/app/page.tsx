'use client'

import { useState, useEffect } from 'react'
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Legend, LineChart, Line
} from 'recharts'
import { 
  Calendar, Users, TrendingUp, AlertTriangle, Clock, 
  ChevronDown, RefreshCw, Download, Building2
} from 'lucide-react'

// API Base URL
const API_URL = 'http://localhost:8000'

// Types
interface Place {
  place_id: number
  total_orders: number
  avg_hourly_orders: number
}

interface HourlyData {
  date: string
  hour: number
  day_name: string
  shift: string
  predicted_orders: number
  staff_needed: number
}

interface DailySummary {
  date: string
  day_name: string
  total_orders: number
  min_staff: number
  max_staff: number
  avg_staff: number
}

interface ForecastData {
  summary: {
    total_orders: number
    avg_staff: number
    peak_staff: number
    days: number
  }
  hourly: HourlyData[]
  daily: DailySummary[]
}

// Card Component
function Card({ children, className = '' }: { children: React.ReactNode, className?: string }) {
  return (
    <div className={`bg-white rounded-xl shadow-sm border border-slate-200 ${className}`}>
      {children}
    </div>
  )
}

// Stat Card Component
function StatCard({ 
  title, 
  value, 
  subtitle, 
  icon: Icon, 
  trend,
  color = 'blue' 
}: { 
  title: string
  value: string | number
  subtitle?: string
  icon: any
  trend?: string
  color?: 'blue' | 'green' | 'purple' | 'orange'
}) {
  const colors = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-emerald-50 text-emerald-600',
    purple: 'bg-purple-50 text-purple-600',
    orange: 'bg-orange-50 text-orange-600',
  }

  return (
    <Card className="p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-slate-500">{title}</p>
          <p className="text-3xl font-bold text-slate-900 mt-1">{value}</p>
          {subtitle && (
            <p className="text-sm text-slate-500 mt-1">{subtitle}</p>
          )}
          {trend && (
            <p className="text-sm text-emerald-600 mt-2 flex items-center gap-1">
              <TrendingUp className="w-4 h-4" />
              {trend}
            </p>
          )}
        </div>
        <div className={`p-3 rounded-lg ${colors[color]}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </Card>
  )
}

// Select Component
function Select({ 
  value, 
  onChange, 
  options,
  label
}: { 
  value: string
  onChange: (val: string) => void
  options: { value: string, label: string }[]
  label: string
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-1">{label}</label>
      <div className="relative">
        <select 
          value={value} 
          onChange={(e) => onChange(e.target.value)}
          className="w-full px-4 py-2.5 bg-white border border-slate-300 rounded-lg appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
        >
          {options.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 pointer-events-none" />
      </div>
    </div>
  )
}

// Slider Component
function Slider({ 
  value, 
  onChange, 
  min, 
  max, 
  label,
  unit = ''
}: { 
  value: number
  onChange: (val: number) => void
  min: number
  max: number
  label: string
  unit?: string
}) {
  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <label className="text-sm font-medium text-slate-700">{label}</label>
        <span className="text-sm font-semibold text-blue-600">{value}{unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
      />
      <div className="flex justify-between text-xs text-slate-400 mt-1">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}

// Main Dashboard Component
export default function Dashboard() {
  const [places, setPlaces] = useState<Place[]>([])
  const [selectedPlace, setSelectedPlace] = useState<string>('')
  const [forecastData, setForecastData] = useState<ForecastData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Settings
  const [forecastDays, setForecastDays] = useState(7)
  const [ordersPerStaff, setOrdersPerStaff] = useState(8)
  const [minStaff, setMinStaff] = useState(2)
  const [maxStaff, setMaxStaff] = useState(15)

  // Disruption handling
  const [showDisruption, setShowDisruption] = useState(false)
  const [disruptionType, setDisruptionType] = useState('call_off')
  const [disruptionShift, setDisruptionShift] = useState('All Day')
  const [callOffs, setCallOffs] = useState(1)
  const [demandIncrease, setDemandIncrease] = useState(30)
  const [disruptionResult, setDisruptionResult] = useState<any>(null)
  const [loadingDisruption, setLoadingDisruption] = useState(false)

  // Smart Alerts
  const [alerts, setAlerts] = useState<any[]>([])
  const [alertsSummary, setAlertsSummary] = useState<any>(null)
  const [showAlerts, setShowAlerts] = useState(false)
  const [loadingAlerts, setLoadingAlerts] = useState(false)

  // Handle disruption recalculation
  const handleRecalculate = async () => {
    if (!selectedPlace) return
    
    setLoadingDisruption(true)
    const today = new Date()
    const dateStr = today.toISOString().split('T')[0]
    
    try {
      const response = await fetch(`${API_URL}/api/replan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          place_id: Number(selectedPlace),
          date: dateStr,
          disruption_type: disruptionType,
          affected_shift: disruptionShift,
          num_calloffs: disruptionType === 'call_off' ? callOffs : 0,
          demand_increase_pct: disruptionType === 'demand_spike' ? demandIncrease : 0,
          orders_per_staff: ordersPerStaff,
          min_staff: minStaff,
          max_staff: maxStaff
        })
      })
      
      const data = await response.json()
      setDisruptionResult(data)
    } catch (err) {
      console.error('Failed to recalculate:', err)
    }
    
    setLoadingDisruption(false)
  }

  // Fetch places on mount
  useEffect(() => {
    fetch(`${API_URL}/api/places`)
      .then(res => res.json())
      .then(data => {
        setPlaces(data)
        if (data.length > 0) {
          setSelectedPlace(String(data[0].place_id))
        }
      })
      .catch(err => setError('Failed to connect to API. Is the server running?'))
  }, [])

  // Fetch forecast when settings change
  useEffect(() => {
    if (!selectedPlace) return
    
    setLoading(true)
    setError(null)
    
    const today = new Date()
    const startDate = today.toISOString().split('T')[0]
    
    fetch(`${API_URL}/api/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        place_id: Number(selectedPlace),
        start_date: startDate,
        days: forecastDays,
        orders_per_staff: ordersPerStaff,
        min_staff: minStaff,
        max_staff: maxStaff
      })
    })
      .then(res => res.json())
      .then(data => {
        setForecastData(data)
        setLoading(false)
      })
      .catch(err => {
        setError('Failed to fetch forecast')
        setLoading(false)
      })

    // Also fetch smart alerts
    setLoadingAlerts(true)
    fetch(`${API_URL}/api/smart-alerts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        place_id: Number(selectedPlace),
        start_date: startDate,
        days: forecastDays,
        orders_per_staff: ordersPerStaff,
        min_staff: minStaff,
        max_staff: maxStaff
      })
    })
      .then(res => res.json())
      .then(data => {
        setAlerts(data.alerts || [])
        setAlertsSummary(data.summary || null)
        setLoadingAlerts(false)
      })
      .catch(err => {
        console.error('Failed to fetch alerts:', err)
        setLoadingAlerts(false)
      })
  }, [selectedPlace, forecastDays, ordersPerStaff, minStaff, maxStaff])

  // Prepare chart data
  const chartData = forecastData?.hourly.map(h => ({
    ...h,
    time: `${h.day_name.slice(0, 3)} ${h.hour}:00`
  })) || []

  const dailyChartData = forecastData?.daily || []

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Calendar className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">Shift Planner</h1>
                <p className="text-xs text-slate-500">AI-Powered Scheduling</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <button 
                onClick={() => setShowDisruption(!showDisruption)}
                className="flex items-center gap-2 px-4 py-2 text-orange-600 bg-orange-50 rounded-lg hover:bg-orange-100 transition-colors"
              >
                <AlertTriangle className="w-4 h-4" />
                <span className="text-sm font-medium">Report Disruption</span>
              </button>
              
              <button className="flex items-center gap-2 px-4 py-2 text-slate-600 bg-slate-100 rounded-lg hover:bg-slate-200 transition-colors">
                <Download className="w-4 h-4" />
                <span className="text-sm font-medium">Export</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error State */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            <p className="font-medium">Error</p>
            <p className="text-sm">{error}</p>
            <p className="text-sm mt-2">Make sure to run: <code className="bg-red-100 px-2 py-0.5 rounded">uvicorn api:app --reload</code></p>
          </div>
        )}

        {/* Controls */}
        <Card className="p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
            <Select
              label="Restaurant"
              value={selectedPlace}
              onChange={setSelectedPlace}
              options={places.map(p => ({
                value: String(p.place_id),
                label: `Place ${p.place_id} (${p.total_orders.toLocaleString()} orders)`
              }))}
            />
            
            <Slider
              label="Forecast Days"
              value={forecastDays}
              onChange={setForecastDays}
              min={1}
              max={14}
              unit=" days"
            />
            
            <Slider
              label="Orders per Staff/Hour"
              value={ordersPerStaff}
              onChange={setOrdersPerStaff}
              min={1}
              max={20}
            />
            
            <Slider
              label="Minimum Staff"
              value={minStaff}
              onChange={setMinStaff}
              min={1}
              max={10}
            />
            
            <Slider
              label="Maximum Staff"
              value={maxStaff}
              onChange={setMaxStaff}
              min={5}
              max={30}
            />
          </div>
        </Card>

        {/* Disruption Panel */}
        {showDisruption && (
          <Card className="p-6 mb-6 border-orange-200 bg-orange-50/50">
            <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-orange-500" />
              Report a Disruption
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Select
                label="Type"
                value={disruptionType}
                onChange={(val) => { setDisruptionType(val); setDisruptionResult(null); }}
                options={[
                  { value: 'call_off', label: 'Staff Call-Off' },
                  { value: 'demand_spike', label: 'Demand Spike' },
                ]}
              />
              
              <Select
                label="Affected Shift"
                value={disruptionShift}
                onChange={(val) => { setDisruptionShift(val); setDisruptionResult(null); }}
                options={[
                  { value: 'All Day', label: 'All Day' },
                  { value: 'Morning', label: 'Morning (6-12)' },
                  { value: 'Afternoon', label: 'Afternoon (12-17)' },
                  { value: 'Evening', label: 'Evening (17-22)' },
                ]}
              />
              
              {disruptionType === 'call_off' ? (
                <Slider
                  label="Staff Calling Off"
                  value={callOffs}
                  onChange={(val) => { setCallOffs(val); setDisruptionResult(null); }}
                  min={1}
                  max={5}
                />
              ) : (
                <Slider
                  label="Demand Increase"
                  value={demandIncrease}
                  onChange={(val) => { setDemandIncrease(val); setDisruptionResult(null); }}
                  min={10}
                  max={100}
                  unit="%"
                />
              )}
              
              <div className="flex items-end">
                <button 
                  onClick={handleRecalculate}
                  disabled={loadingDisruption}
                  className="w-full px-4 py-2.5 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {loadingDisruption ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Calculating...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="w-4 h-4" />
                      Recalculate Schedule
                    </>
                  )}
                </button>
              </div>
            </div>
            
            {/* Disruption Results */}
            {disruptionResult && (
              <div className="mt-6 p-4 bg-white rounded-lg border border-orange-200">
                <h4 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                  <span className="text-green-500">💡</span> AI Recommendation
                </h4>
                
                <div className="p-3 bg-green-50 rounded-lg border border-green-200 mb-4">
                  <p className="text-green-800 font-medium">{disruptionResult.recommendation}</p>
                </div>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="p-3 bg-slate-50 rounded-lg">
                    <p className="text-sm text-slate-500">Disruption Type</p>
                    <p className="font-semibold text-slate-900 capitalize">
                      {disruptionResult.disruption_type === 'call_off' ? '👤 Staff Call-Off' : '📈 Demand Spike'}
                    </p>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-lg">
                    <p className="text-sm text-slate-500">Affected Shift</p>
                    <p className="font-semibold text-slate-900">{disruptionResult.affected_shift}</p>
                  </div>
                </div>
                
                {disruptionResult.adjusted_hours && disruptionResult.adjusted_hours.length > 0 && (
                  <>
                    <h5 className="font-medium text-slate-700 mb-2">Staffing Adjustments Needed:</h5>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {disruptionResult.adjusted_hours.map((hour: any, idx: number) => (
                        <div 
                          key={idx}
                          className="flex items-center justify-between p-2 bg-slate-50 rounded"
                        >
                          <span className="text-sm text-slate-600">
                            {hour.hour}:00 - {hour.shift}
                          </span>
                          <div className="flex items-center gap-3">
                            <span className="text-sm text-slate-500">
                              {hour.original_staff} → {hour.new_staff} staff
                            </span>
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                              hour.change > 0 
                                ? 'bg-red-100 text-red-700' 
                                : 'bg-green-100 text-green-700'
                            }`}>
                              {hour.change > 0 ? '+' : ''}{hour.change}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                      <p className="text-sm text-blue-800">
                        <strong>Total Impact:</strong> {disruptionResult.total_additional_staff_hours} additional staff-hours needed
                        {disruptionResult.total_additional_staff_hours > 0 && (
                          <span className="ml-2 text-blue-600">
                            (Est. cost: ${disruptionResult.total_additional_staff_hours * 15})
                          </span>
                        )}
                      </p>
                    </div>
                  </>
                )}
                
                {disruptionResult.adjusted_hours && disruptionResult.adjusted_hours.length === 0 && (
                  <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                    <p className="text-sm text-green-800">
                      ✅ Current staffing levels can handle this disruption. No additional staff needed.
                    </p>
                  </div>
                )}
              </div>
            )}
          </Card>
        )}

        {/* Stats */}
        {forecastData && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
            <StatCard
              title="Total Predicted Orders"
              value={forecastData.summary.total_orders.toLocaleString()}
              subtitle={`Next ${forecastData.summary.days} days`}
              icon={TrendingUp}
              color="blue"
            />
            <StatCard
              title="Average Staff Needed"
              value={forecastData.summary.avg_staff}
              subtitle="Per hour"
              icon={Users}
              color="green"
            />
            <StatCard
              title="Peak Staff Required"
              value={forecastData.summary.peak_staff}
              subtitle="Maximum at any hour"
              icon={Users}
              color="purple"
            />
            <div onClick={() => setShowAlerts(!showAlerts)} className="cursor-pointer">
              <StatCard
                title="Staffing Alerts"
                value={alertsSummary?.total_alerts || 0}
                subtitle={alertsSummary?.high_severity > 0 ? `${alertsSummary.high_severity} high priority` : "Click to view details"}
                icon={AlertTriangle}
                color="orange"
              />
            </div>
          </div>
        )}

        {/* Smart Alerts Panel */}
        {showAlerts && alerts.length > 0 && (
          <Card className="p-6 mb-6 border-orange-200">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-orange-500" />
                AI-Powered Smart Alerts & Recommendations
              </h3>
              <button 
                onClick={() => setShowAlerts(false)}
                className="text-slate-400 hover:text-slate-600"
              >
                ✕
              </button>
            </div>
            
            {/* Alert Summary */}
            {alertsSummary && (
              <div className="grid grid-cols-4 gap-4 mb-6 p-4 bg-slate-50 rounded-lg">
                <div className="text-center">
                  <p className="text-2xl font-bold text-slate-900">{alertsSummary.total_alerts}</p>
                  <p className="text-sm text-slate-500">Total Alerts</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-red-600">{alertsSummary.high_severity}</p>
                  <p className="text-sm text-slate-500">High Priority</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-orange-500">{alertsSummary.medium_severity}</p>
                  <p className="text-sm text-slate-500">Medium Priority</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-slate-900">{alertsSummary.estimated_labor_hours}</p>
                  <p className="text-sm text-slate-500">Est. Labor Hours</p>
                </div>
              </div>
            )}
            
            {/* Alerts List */}
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {alerts.map((alert: any) => (
                <div 
                  key={alert.id}
                  className={`p-4 rounded-lg border-l-4 ${
                    alert.severity === 'high' 
                      ? 'bg-red-50 border-red-500' 
                      : alert.severity === 'medium'
                      ? 'bg-orange-50 border-orange-500'
                      : 'bg-blue-50 border-blue-500'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          alert.severity === 'high' 
                            ? 'bg-red-100 text-red-700' 
                            : alert.severity === 'medium'
                            ? 'bg-orange-100 text-orange-700'
                            : 'bg-blue-100 text-blue-700'
                        }`}>
                          {alert.severity.toUpperCase()}
                        </span>
                        <span className="text-xs text-slate-500">
                          {alert.day_name} {alert.date} {alert.hour !== null ? `@ ${alert.hour}:00` : ''}
                        </span>
                      </div>
                      
                      <h4 className="font-semibold text-slate-900">{alert.title}</h4>
                      <p className="text-sm text-slate-600 mt-1">{alert.description}</p>
                      
                      {/* Recommendation Box */}
                      <div className="mt-3 p-3 bg-white rounded-lg border border-slate-200">
                        <p className="text-sm font-medium text-slate-900 flex items-center gap-2">
                          <span className="text-green-500">💡</span> AI Recommendation
                        </p>
                        <p className="text-sm text-slate-700 mt-1">{alert.recommendation.message}</p>
                        
                        {alert.recommendation.cost_impact && (
                          <p className="text-xs text-slate-500 mt-2 flex items-center gap-1">
                            <span>💰</span> {alert.recommendation.cost_impact}
                          </p>
                        )}
                        
                        {alert.recommendation.risk_if_ignored && (
                          <p className="text-xs text-red-600 mt-1 flex items-center gap-1">
                            <span>⚠️</span> Risk: {alert.recommendation.risk_if_ignored}
                          </p>
                        )}
                      </div>
                    </div>
                    
                    {/* Action Button */}
                    <div className="ml-4">
                      <button className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                        alert.recommendation.action === 'increase_staff'
                          ? 'bg-green-100 text-green-700 hover:bg-green-200'
                          : alert.recommendation.action === 'reduce_staff'
                          ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                          : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                      }`}>
                        {alert.recommendation.action === 'increase_staff' && `+${alert.recommendation.additional_staff} Staff`}
                        {alert.recommendation.action === 'reduce_staff' && `-${alert.recommendation.reduction} Staff`}
                        {alert.recommendation.action === 'prepare_weekend' && 'Plan Weekend'}
                        {alert.recommendation.action === 'prepare_lunch' && 'Plan Lunch'}
                        {alert.recommendation.action === 'prepare_dinner' && 'Plan Dinner'}
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* No Alerts Message */}
        {showAlerts && alerts.length === 0 && (
          <Card className="p-6 mb-6 border-green-200 bg-green-50">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-100 rounded-full">
                <Users className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <h3 className="font-semibold text-green-900">All Clear!</h3>
                <p className="text-sm text-green-700">No staffing alerts for the forecast period. Current staffing levels appear adequate.</p>
              </div>
            </div>
          </Card>
        )}

        {/* Charts */}
        {forecastData && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            {/* Main Chart */}
            <Card className="lg:col-span-2 p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4">Demand Forecast</h3>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id="colorOrders" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis 
                      dataKey="time" 
                      tick={{ fontSize: 12 }} 
                      stroke="#94a3b8"
                      interval={5}
                    />
                    <YAxis tick={{ fontSize: 12 }} stroke="#94a3b8" />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'white', 
                        border: '1px solid #e2e8f0',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'
                      }}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="predicted_orders" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      fillOpacity={1} 
                      fill="url(#colorOrders)"
                      name="Predicted Orders"
                    />
                    <Line 
                      type="stepAfter" 
                      dataKey="staff_needed" 
                      stroke="#a855f7" 
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={false}
                      name="Staff Needed"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* Daily Summary */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4">Daily Summary</h3>
              <div className="space-y-3">
                {forecastData.daily.map((day) => (
                  <div 
                    key={day.date} 
                    className="flex items-center justify-between p-3 bg-slate-50 rounded-lg"
                  >
                    <div>
                      <p className="font-medium text-slate-900">{day.day_name}</p>
                      <p className="text-sm text-slate-500">{day.date}</p>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold text-slate-900">{day.total_orders} orders</p>
                      <p className="text-sm text-slate-500">{day.min_staff}-{day.max_staff} staff</p>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* Staffing by Day Chart */}
        {forecastData && (
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Staffing by Day</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dailyChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="day_name" tick={{ fontSize: 12 }} stroke="#94a3b8" />
                  <YAxis tick={{ fontSize: 12 }} stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'white', 
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  <Bar dataKey="total_orders" fill="#3b82f6" name="Total Orders" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="max_staff" fill="#a855f7" name="Max Staff" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-8 h-8 text-blue-500 animate-spin" />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-slate-500">
            Shift Planning Decision Assistant • Built for Hackathon 2024
          </p>
        </div>
      </footer>
    </div>
  )
}
