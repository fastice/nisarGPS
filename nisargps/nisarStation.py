#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:31:25 2024

@author: ian
"""
import functools
import numpy as np
import calendar
from datetime import timedelta, datetime
import os
import pyproj
from scipy.stats import linregress
import nisarcryodb
import sys
import traceback

class nisarStation():
    '''
    Abstract class to define parser for NISAR HDF products.
    '''
 
            
    def catchError(func):
        ''' Decorator to trap and abbreviate errors '''
        @functools.wraps(func)
        def catchErrorInner(inst, *args, **kwargs):
            traceBack = False
            if 'traceBack' in kwargs:
                traceBack = kwargs['traceBack']
            try:
                return func(inst, *args, **kwargs)
            except Exception as errMsg:
                #print(traceback.format_exc())
                excType, excObj, excTb = sys.exc_info()
                tb = excTb
                while tb is not None:
                    line = tb.tb_lineno
                    tb = tb.tb_next
                msg = f'Error in: {type(inst).__name__}.{func.__name__} at ' \
                    f'line {line} \nMessage: {errMsg}'
                inst.printError(msg)
                if not traceBack:
                    sys.tracebacklimit=0
                
                def myExit(traceBack):
                    try:
                        sys.exit()
                    except SystemExit as message:
                        if traceBack:
                            sys.exit()
                # This will only do a full exit if traceBack is True
                myExit(traceBack)
        return catchErrorInner
  

    
    @catchError
    def __init__(self, stationName,  epsg=None, useDB=True, DBConnection=None,
                 DBConfigFile='calvaldb_config.ini', **kwargs):
        '''
        Class for handling GPS data for a NISAR station

        Parameters
        ----------
        stationName : str
            Four character station ID.
        epsg : str, optional
            Epsg code for for Arctic (3413) or Antarctic (3031). Default is None to autodetect.
        useDB : bool, optional
            Get GPS from database. Altertnatively read from text file (not fully support). Default is True.
        DBConnection: nisarcryodb
            Can pass in a connection, which allows a connection to be use for multiple stations. 
            The default None so new connection is opened.
        DBConfigFile: str, optional
            Config file path. Default is ./calvaldb_config.ini. 
        traceBack: bool, optional
            By default error handler prints abrv. msg. Set true for full traceback.
        **keywords : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        print(DBConnection)
        self.stationName = stationName
        self.date = np.array([])
        self.epoch = np.array([])
        self.lat = np.array([])
        self.lon = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])
        self.sigma3 = np.array([])
        #
        self.epsg = epsg
        self.lltoxy = None
        #
        if useDB and DBConnection is None:
            self.DB = nisarcryodb.nisarcryodb(configFile=DBConfigFile)
        elif DBConnection is not None:
            self.DB = DBConnection
        else:
            self.DB = None
        # Set station id if using a data base
        self.stationID = None
        if self.DB is not None:
             self.stationID = self.DB.stationNameToID(stationName)
        # Cache year lengths for speed
        self._computeYearLengthLookUp(**kwargs)

 
    
    @catchError       
    def _computeYearLengthLookUp(self, **kwargs):
        '''
        Cache year lengths to speedup date conversions
        '''
        years = range(1990, 2100)
        lengths = np.array([(datetime(y + 1, 1, 1) - 
                    datetime(y, 1, 1)).total_seconds() for y in years])
        self.yearLengthLookupSeconds = dict(zip(years, lengths))
        self.yearLengthLookupDays = dict(zip(years, lengths/86400))
       
    def printError(self, msg):
        '''
        Print error message
        Parameters
        ----------
        msg : str
            error message.
        Returns
        -------
        None
        '''
        length = max([len(x) for x in msg.split('\n')])
        stars = ''.join(['*']*length)
        print(f'\n\033[1;31m{stars}\n{msg} \n{stars}\n \033[0m\n')
    
    @catchError
    def _initCoordinateConversion(self, **kwargs):
        print('EPSG', self.epsg)
        self.crs = pyproj.CRS.from_epsg(str(self.epsg))
        self.proj = pyproj.Proj(str(self.epsg))
        if self.lltoxy is None:
            self.lltoxy = pyproj.Transformer.from_crs("EPSG:4326",
                                                      f"EPSG:{self.epsg}"
                                                      ).transform
    
    @catchError
    def _determineEPSG(self, lat, **kwargs):
        '''
        Determine epsg base on lat (3031, 3413) based on a lat value
        lat: float
            latitude value used to setepsg.
        Returns
        -------
        None.

        '''
        if self.epsg is None:
            if lat < -55:
                self.epsg = 3031
            elif lat > 55:
                self.epsg = 3413
        #

    @catchError
    def _readFile(self, filePath, **kwargs):
        '''
        Read a JPL processed GPS file

        Parameters
        ----------
        filePath : str
            Path to GPS file.

        Returns
        -------
        6xN result with date, decDate, lat, lon, z, sigma

        '''
        if not os.path.exists(filePath):
            self.printError(f'Cannot open {filePath}')
        newData = []
        date = []
        count = 0
        with open(filePath) as fpGPS:
            for line in fpGPS:
                # Process line
                pieces = line.split()
                if len(pieces) != 6 or self.stationName != pieces[-1].strip():
                    print(f'skipping line {count} missing data or invalid '
                          'station')
                # Grab data
                lineData = [float(x) for x in pieces[0:-1]]
                # compute datetime
                year = int(lineData[0])
                sec = np.rint((lineData[0] - year) *
                              (365 + int(calendar.isleap(year))) * 86400)
                date.append(datetime(year, 1, 1, 0, 0, 0) +
                            timedelta(seconds=sec))
                newData.append(lineData)
                count += 1
        # returns date, epoch, lat, lon, z, sigmax
        epoch, lat, lon, z, sigma = np.transpose(newData)

        return np.array(date),  epoch, lat, lon, z, sigma
   
    @catchError
    def addData(self, filePath, **kwargs):
        '''
         Read data and merge with any existing data

        Parameters
        ----------
        filePath : str
            Path to GPS file.

        Returns
        -------
        None.

        '''
        date, epoch, lat, lon, z, sigma3 = self._readFile(filePath)
        #
        self._determineEPSG(lat[0])
        #
        x, y = self.lltoxy(lat, lon)
        # add data
        for var, data in zip(
                ['date', 'epoch',  'lat', 'lon', 'x', 'y', 'z', 'sigma3'],
                [date, epoch, lat, lon, x, y, z, sigma3]):
            setattr(self, var, np.append(data, getattr(self, var)))
        #

        # Now make sure all monotonic in time
        sortOrder = np.argsort(self.epoch)
        for var, data in zip(['date', 'epoch',  'lat', 'lon', 'x', 'y', 'z',
                              'sigma3'],
                             [date, epoch, lat, lon, x, y, z, sigma3]):
            setattr(self, var, getattr(self, var)[sortOrder])
        #
        self.meanLat = np.mean(lat)
        #
        self.projLengthScale = self.proj.get_factors(0, self.meanLat
                                                     ).parallel_scale


    @catchError
    def computeVelocity(self, date1, date2, method='regression', minPoints=10,
                        dateFormat='%Y-%m-%d', **kwargs):
        '''
         Compute velocity for date range

        Parameters
        ----------
        date1 : datetime date
            First date in interval to compute date.
        date2 : datetime date
            Second date in interval to compute date.
        method : str, optional
            Use either point or regression. The default is 'regression'
        dateFormat : TYPE, optional
            DESCRIPTION. The default is '%Y-%m-%d'.
        minPoints : TYPE, optional
            Return nan's if # of valid points is < minPoits. The default is 10.
        averagingPeriod : number, optional
            For 'point' methdod only. The number of hours on either side of 
            date to average when computeing two point positions.
            The default is 12.
        Returns
        -------
        vx, vy, x, y : velocity and mean location of measurement

        '''
        if method == 'regression':
            return self.computeVelocityRegression(date1, date2,
                                                  minPoints=minPoints,
                                                  dateFormat=dateFormat,
                                                  **kwargs)
        elif method == 'point':
            return self.computeVelocityPtToPt(date2, date1, date2,
                                              minPoints=minPoints,
                                              dateFormat=dateFormat,
                                              averagingPeriod=averagingPeriod,
                                              **kwargs)
        else:
            self.printError(f'Invalid method {method}, use point or regression')
            return np.nan, np.nan, np.nan, np.nan
         
    @catchError
    def computeVelocityRegression(self, date1, date2, minPoints=10,
                        dateFormat='%Y-%m-%d',  **kwargs):
        '''
         Compute velocity for date range

        Parameters
        ----------
        date1 : datetime date
            First date in interval to compute date.
        date2 : datetime date
            Second date in interval to compute date..
        dateFormat : TYPE, optional
            DESCRIPTION. The default is '%Y-%m-%d'.
        minPoints : TYPE, optional
            Return nan's if # of valid points is < minPoits. The default is 10.
        Returns
        -------
        vx, vy, x, y : velocity and mean location of measurement

        '''
        date, x, y, z, epoch = self.subsetXYZ(date1, date2,
                                              dateFormat=dateFormat, **kwargs)
        if x is np.nan:
            return np.nan, np.nan
        #
        # Uses slope of linear regression as velocity estimate
        vxPS, intercept, rx, px, sigmax = linregress(epoch, x)
        vyPS, intercept, ry, py, sigmay = linregress(epoch, y)
        # Scale from projected to actual coordinates
        return vxPS/self.projLengthScale, vyPS/self.projLengthScale, np.mean(x), np.mean(y)
        
    @catchError
    def computeVelocityPtToPt(self, date1, date2, minPoints=10,
                              dateFormat='%Y-%m-%d', averagingPeriod=12, **kwargs):
        '''
         Compute velocity for date range differencing point positions

        Parameters
        ----------
        date1 : datetime date
            First date in interval to compute date.
        date2 : datetime date
            Second date in interval to compute date.
        dateFormat : TYPE, optional
            DESCRIPTION. The default is '%Y-%m-%d'.
        minPoints : int, optional
            Return nan's if # of valid points is < minPoits. The default is 10.
        averagingPeriod : number, optional
            Number of hours on either side of date to average the positions.
            The default is 12.
        Returns
        -------
        None.

        '''
        date1 = self._formatDate(date1, dateFormat=dateFormat)
        date2 = self._formatDate(date2, dateFormat=dateFormat)
        tAvg = timedelta(hours=averagingPeriod)
        #
        dates1, x1, y1, z1, epoch1 = self.subsetXYZ(date1 - tAvg,
                                                    date1 + tAvg,
                                                    minPoints=minPoints,
                                                    **kwargs)
        dates2, x2, y2, z2, epoch2 = self.subsetXYZ(date2 - tAvg,
                                                    date2 + tAvg,
                                                    minPoints=minPoints,
                                                    **kwargs)
        if x1 is np.nan or x2 is np.nan:
            return np.nan, np.nan
        #
        # Uses slope of linear regression as velocity estimate
        x1Avg, x2Avg = np.mean(x1), np.mean(x2)
        y1Avg, y2Avg = np.mean(y1), np.mean(y2)

        epoch1Avg, epoch2Avg = np.mean(epoch1), np.mean(epoch2)
        dT = epoch2Avg - epoch1Avg
        #
        vxPS = (x2Avg - x1Avg) / dT
        vyPS = (y2Avg - y1Avg) / dT
        xPos, yPos = (x1Avg + x2Avg) * 0.5
        yPos, yPos = (y1Avg + y2Avg) * 0.5
        # Scale from projected to actual coordinates
        return vxPS/self.projLengthScale, vyPS/self.projLengthScale, xPos, yPos

    @catchError
    def computeVelocityTimeSeries(self, date1, date2, dT, sampleInterval,
                                  dateFormat='%Y-%m-%d', method='regression',
                                  averagingPeriod=None, minPoints=10,  **kwargs):
        '''
        Compute velocity time series from JPL data

        Parameters
        ----------
        date1 : str or datetime
            First date in time series.
        date2 : str or datetime
            Last date in time series.
        dT : number
            Delta time for computing speed. (in hours)
        sampleInterval : number
            Frequency at which to compute estimates (hours).
        method : str, optional
            Use either point or regression. The default is 'regression'
        dateFormat : str, optional
            If date1/2 is a str, to datetime format. The default is '%Y-%m-%d'.
        method 
        Returns
        -------
        vx, vy: nparray
            velocty time series with samples every sampleInterval hours.

        '''
        if method not in ['point', 'regression']:
            self.printError(f'Invalid method {method} keyword, must be point'
                            ' or regression')
        if method == 'point':
            if averagingPeriod is None:
                averagingPeriod = dT/24.
        #
        # Convert to datetime if needed
        date1 = self._formatDate(date1, dateFormat=dateFormat)
        date2 = self._formatDate(date2, dateFormat=dateFormat)
        # Initialize
        currentDate = date1
        vxSeries, vySeries, dateSeries, xSeries, ySeries = [], [], [], [], []
        #
        # Loop to compute velocities at sample intervale.
        while currentDate + timedelta(hours=dT) < date2:
           
            vx, vy, x, y = self.computeVelocity(currentDate,
                                                currentDate + timedelta(hours=dT),
                                                method=method,
                                                minPoints=minPoints,
                                                averagingPeriod=averagingPeriod
                                               )
            #
            dateSeries.append(currentDate + timedelta(hours=dT/2))
            vxSeries.append(vx)
            vySeries.append(vy)
            xSeries.append(x)
            ySeries.append(y)
            currentDate = currentDate + timedelta(hours=sampleInterval)
        return np.array(dateSeries), np.array(vxSeries), np.array(vySeries), np.array(xSeries), np.array(ySeries)


    @catchError
    def _formatDate(self, date, dateFormat='%Y-%m-%d', **kwargs):
        '''
        Format dates as str to datetime

        Parameters
        ----------
        date : str or datetime
            date

        dateFormat : str, optional
            If date1/2 is a str, to datetime format. The default is '%Y-%m-%d'.

        Returns
        -------
        date as datetime.

        '''
        if type(date) is str:
            return datetime.strptime(date, dateFormat)
        return date

    @catchError
    def _datetimeToDecimalYear(self, date, **kwargs):
        '''
        Convert date time to decimal year
        
        Parameters
        ----------
        date : datetime
            date
        '''
        year = date.year
        yearLength = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).total_seconds()
        return year + (date - datetime(year, 1, 1)).total_seconds() / yearLength

    @catchError
    def _DecimalYearToDatetime(self, date, **kwargs):
        '''
        Convert date time to decimal year
        
        Parameters
        ----------
        date : datetime
            date
        '''
        year = int(date)
        fracYear = date - year
        yearLength = self.yearLengthLookupSeconds[year]
        return datetime(year, 1, 1) + timedelta(seconds=fracYear * yearLength)

    @catchError
    def _DecimalYearToDOYVector(self, date, **kwargs):
        '''
        Convert decimal year to doy
        
        Parameters
        ----------
        date : datetime
            date
            
        '''
        # Compute year, frac year, and length of year
        year = date.astype(int)
        fracYear = date - year
        yearLength = np.array([self.yearLengthLookupDays[y] for y in year])
        # Compute doy
        return (fracYear * yearLength).astype(int) + 1

    @catchError
    def subsetXYZ(self, date1, date2, dateFormat='%Y-%m-%d %H:%M:%S', minPoints=1, quiet=True, removeOverlap=True, **kwargs):
        '''
        Return all x,y, z points in interval [date1, date2]

        Parameters
        ----------
        date1 : str or datetime
            First date datetime or ascii with formate specified by dateFormat.
        date2 : str or datetime
            Last date datetime or ascii with formate specified by dateFormat.
        dateFormat : datetime format str, optional
            Format for conversion to date time. The default is '%Y-%m-%d %H:%M:%S'.
        minPoints : int, optional
            Return nan's if # of valid points is < minPoits. The default is 1.
        removeOverlap : bool, optional
            Return nan's if # of valid points is < minPoits. The default is 1.
        sigmaMultiple : int, optional.
            Discard outliers > sigmaMuliple * sigma. Use None for no removal. The defaul is 3.
        Returns
        -------
        x, y, z np.array
            x, y, z values in projected coordinates.

        '''
        # Convert to datetime if needed
        date1 = self._formatDate(date1, dateFormat=dateFormat, **kwargs)
        date2 = self._formatDate(date2, dateFormat=dateFormat, **kwargs)
        #
        if self.DB is None:
            return self._subsetXYZtext(date1, date2, minPoints=minPoints, quiet=quiet, **kwargs)
        return self._subsetXYZDB(date1, date2, minPoints=minPoints, quiet=quiet, **kwargs)

    @catchError
    def _subsetXYZDB(self, date1, date2, dateFormat='%Y-%m-%d %H:%M:%S', 
                     minPoints=1, removeOverlap=True, sigmaMultiple=3, quiet=True, **kwargs):
        '''
        Return all x,y, z points in interval [date1, date2]

        Parameters
        ----------
        date1 : str or datetime
            First date datetime or ascii with formate specified by dateFormat.
        date2 : str or datetime
            Last date datetime or ascii with formate specified by dateFormat.
        dateFormat : datetime format str, optional
            Format for conversion to date time. The default is '%Y-%m-%d %H:%M:%S'.
        minPoints : int, optional
            Return nan's if # of valid points is < minPoits. The default is 1.
        removeOverlap : bool, optional
            Return nan's if # of valid points is < minPoits. The default is 1.
        sigmaMultiple : int, optional.
            Discard outliers > sigmaMuliple * sigma. Use None for no removal. The defaul is 3.
        Returns
        -------
        x, y, z np.array
            x, y, z values in projected coordinates.

        '''
        # Use decimal dates for DB queery
        d1 = self._datetimeToDecimalYear(date1, **kwargs)
        d2 = self._datetimeToDecimalYear(date2, **kwargs)
  
        # Query data base for station data
        data = self.DB.getStationDateRangeData(self.stationName, d1, d2, 'landice', 'gps_data')
        # This removes the overlap that comes with the GPS day files
        if removeOverlap:
            data = self._removeOverlap(data)
        # Check if there is data
        if len(data) == 0:
            if not quiet:
                print(f'no data in date range for {self.stationName}')
            return np.nan, np.nan, np.nan, np.nan, np.nan
        # Make sure all is in place for coordinate conversion
        if self.epsg is None:
            self._determineEPSG(data['lat'].to_numpy()[0])
        if self.lltoxy is None:
            self._initCoordinateConversion()
        # Convert to x, y
        x, y = self.lltoxy(data['lat'].to_numpy(), data['lon'].to_numpy())
        #
        if sigmaMultiple is not None:
            x, y, data = self.removeOutliers(x, y, data,
                                             sigmaMultiple=sigmaMultiple, **kwargs)
        #
        date = [self._DecimalYearToDatetime(d) for d in data['decimal_year']]
        # Lat dependendent length scale for projected meters to real meters
        self.meanLat = np.mean(data['lat'].to_numpy())
        self.projLengthScale = self.proj.get_factors(0, self.meanLat
                                                     ).parallel_scale
        return date, x, y, data['ht_abv_eps'].to_numpy(), data['decimal_year'].to_numpy()

    @catchError
    def removeOutliers(self, x, y, data, sigmaMultiple=3, **kwargs):
        '''
        Remove data where date is not equal to the nominal doy to remove overlap
              date1 : TYPE
            DESCRIPTION.
        '''
        epoch = data['decimal_year'].to_numpy()
        good = np.ones(x.shape, dtype=bool)  # Initially keep all
        for d in [x, y, data['ht_abv_eps'].to_numpy()]:
            fitResult = linregress(epoch, d)
            fit = fitResult[0] * epoch + fitResult[1]
            detrended = d - fit
            sigma = np.std(detrended)
            good = np.logical_and(good, np.abs(detrended) < sigmaMultiple*sigma)
        return x[good], y[good], data[good]

    @catchError
    def _removeOverlap(self, data, **kwargs):
        '''
        Remove data where date is not equal to the nominal doy to remove overlap
        
        Parameters
        ----------
        date1 : str or datetime
            First date datetime or ascii with formate specified by dateFormat.
        date2 : str or datetime
            Last date datetime or ascii with formate specified by dateFormat.
        dateFormat : datetime format str, optional
            Format for conversion to date time. The default is '%Y-%m-%d %H:%M:%S'.
        minPoints : int, optional
            Return nan's if # of valid points is < minPoits. The default is 1.
        removeOverlap : bool, optional
            Return nan's if # of valid points is < minPoits. The default is 1.
        sigmaMultiple : int, optional.
            Discard outliers > sigmaMuliple * sigma. Use None for no removal. The defaul is 3.
        '''
        doy = self._DecimalYearToDOYVector(data['decimal_year'].to_numpy())
        nominalDoi = data['nominal_doy'].to_numpy()
        keep = doy == nominalDoi
        return data[keep]

    @catchError
    def _subsetXYZtext(self, date1, date2, minPoints=1, quiet=True, **kwargs):
        '''
        Return all x, y, z points in interval [date1, date2] for text input

        Parameters
        ----------
        date1 : TYPE
            DESCRIPTION.
        date2 : TYPE
            DESCRIPTION.
        minPoints : TYPE, optional
            Return nan's if # of valid points is < minPoits. The default is 1.

        Returns
        -------
        x, y, z np.array
            x, y, z values in projected coordinates.

        '''
        inRange = np.logical_and(self.date >= date1, self.date <= date2)
        if inRange.sum() < minPoints:
            if not quiet:
                print(f'no data in date range for {self.stationName}')
            return np.nan, np.nan, np.nan
        #
        return self.date[inRange], self.x[inRange], self.y[inRange], \
            self.z[inRange], self.epoch[inRange]

