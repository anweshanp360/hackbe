// db/config.js
const sql = require('mssql');

const config = {
  server: 'DESKTOP-3LI0PBS',
  database: 'MyAppDB',
  user: 'trialAppUserNew',
  password: 'Password123!',
  port:1433,
  options: {
    trustServerCertificate: true, // for self-signed certs
    encrypt: true,               // set to false for local
  },
};



const poolPromise = new sql.ConnectionPool(config)
  .connect()
  .then(pool => {
    console.log('✅ Connected to MSSQL using tedious');
    return pool;
  })
  .catch(err => {
    console.error('❌ Database Connection Failed:', err);
    throw err;
  });

module.exports = {
  sql,
  poolPromise,
};
