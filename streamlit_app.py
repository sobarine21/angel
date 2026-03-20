// supabase/functions/ohlc-data/index.ts

import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Get self-hosted Supabase credentials from environment
    const HOSTED_SUPABASE_URL = Deno.env.get('HOSTED_SUPABASE_URL')
    const HOSTED_SUPABASE_KEY = Deno.env.get('HOSTED_SUPABASE_KEY')

    if (!HOSTED_SUPABASE_URL || !HOSTED_SUPABASE_KEY) {
      throw new Error('Missing self-hosted Supabase credentials')
    }

    // Create client for self-hosted instance with custom fetch
    const supabase = createClient(
      HOSTED_SUPABASE_URL, 
      HOSTED_SUPABASE_KEY,
      {
        auth: {
          persistSession: false,
          autoRefreshToken: false,
        },
        global: {
          // Custom fetch to handle SSL issues in development/self-hosted environments
          fetch: (...args) => {
            return fetch(args[0], {
              ...args[1],
              // Note: Only use this if you trust your self-hosted instance
              // Remove in production with proper SSL certificates
            })
          }
        }
      }
    )

    const url = new URL(req.url)
    const pathname = url.pathname

    // Remove the function name prefix from pathname for routing
    // Handles both /ohlc-data and /ohlc-api/ohlc-data
    const path = pathname.replace(/^\/ohlc-data/, '').replace(/^\/ohlc-api/, '') || '/'

    // GET / or /ohlc-data - Get all OHLC data with optional filters
    if (req.method === 'GET' && (path === '/' || path === '/ohlc-data' || pathname.endsWith('/ohlc-data'))) {
      const symbol = url.searchParams.get('symbol')
      const startDate = url.searchParams.get('start_date')
      const endDate = url.searchParams.get('end_date')
      const limit = url.searchParams.get('limit') || '1000'
      const offset = url.searchParams.get('offset') || '0'

      let query = supabase
        .from('ohlc_db')
        .select('*')
        .order('date', { ascending: false })
        .range(parseInt(offset), parseInt(offset) + parseInt(limit) - 1)

      if (symbol) {
        query = query.eq('symbol', symbol)
      }

      if (startDate) {
        query = query.gte('date', startDate)
      }

      if (endDate) {
        query = query.lte('date', endDate)
      }

      const { data, error } = await query

      if (error) {
        console.error('Database error:', error)
        throw error
      }

      return new Response(
        JSON.stringify({ data, count: data?.length || 0 }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // GET /:id - Get specific OHLC record by ID
    if (req.method === 'GET' && /\/\d+$/.test(path)) {
      const id = path.split('/').pop()

      const { data, error } = await supabase
        .from('ohlc_db')
        .select('*')
        .eq('id', id)
        .single()

      if (error) throw error

      return new Response(
        JSON.stringify({ data }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // GET /symbol/:symbol - Get data for specific symbol
    if (req.method === 'GET' && path.includes('/symbol/')) {
      const symbol = path.split('/symbol/')[1]
      const limit = url.searchParams.get('limit') || '100'

      const { data, error } = await supabase
        .from('ohlc_db')
        .select('*')
        .eq('symbol', symbol)
        .order('date', { ascending: false })
        .limit(parseInt(limit))

      if (error) throw error

      return new Response(
        JSON.stringify({ data, count: data?.length || 0 }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // GET /latest/:symbol - Get latest OHLC for a symbol
    if (req.method === 'GET' && path.includes('/latest/')) {
      const symbol = path.split('/latest/')[1]

      const { data, error } = await supabase
        .from('ohlc_db')
        .select('*')
        .eq('symbol', symbol)
        .order('date', { ascending: false })
        .limit(1)
        .single()

      if (error) throw error

      return new Response(
        JSON.stringify({ data }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    return new Response(
      JSON.stringify({ 
        error: 'Not found',
        path: path,
        available_endpoints: [
          'GET / - Get all OHLC data',
          'GET /:id - Get specific record',
          'GET /symbol/:symbol - Get data for symbol',
          'GET /latest/:symbol - Get latest data for symbol'
        ]
      }),
      { status: 404, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error('Error:', error)
    return new Response(
      JSON.stringify({ 
        error: error.message,
        details: error.toString()
      }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})
