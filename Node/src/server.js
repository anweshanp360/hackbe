const express = require('express');
const cors = require('cors');
const patientRoutes = require('./routes/patient.routes');
const trialRoutes = require('./routes/trial.routes');
const { poolPromise } = require('./db/config');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/patients', patientRoutes);
app.use('/api/trials', trialRoutes);

// ✅ DB connection test at server start using poolPromise
poolPromise
  .then(() => {
    console.log('✅ SQL Server connected successfully');
    app.listen(PORT, () => {
      console.log(`🚀 Server running at http://localhost:${PORT}`);
    });
  })
  .catch((err) => {
    console.error('❌ SQL Server connection failed at startup:', err);
  });
