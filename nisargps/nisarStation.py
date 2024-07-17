#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:31:25 2024

@author: ian
"""
import numpy as np
import calendar
from datetime import timedelta, datetime
import os
import pyproj
from scipy.stats import linregress


class nisarStation():
    '''
    Abstract class to define parser for NISAR HDF products.
    '''

#    __metaclass__ = ABCMeta

    def __init__(self, stationID,  epsg=None):
        '''
        Class for handling GPS data for a NISAR station

        Parameters
        ----------
        stationID : str
            Four character station ID.
        **keywords : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.stationID = stationID
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
        print(f'\n\t\033[1;31m *** {msg} *** \033[0m\n')

    def _determineEPSG(self, lat):
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
            elif self.lat > 55:
                self.epsg = 3413
        #
        self.crs = pyproj.CRS.from_epsg(str(self.epsg))
        self.proj = pyproj.Proj(str(self.epsg))
        if self.lltoxy is None:
            self.lltoxy = pyproj.Transformer.from_crs("EPSG:4326",
                                                      f"EPSG:{self.epsg}"
                                                      ).transform

    def _readFile(self, filePath):
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
                if len(pieces) != 6 or self.stationID != pieces[-1].strip():
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

    def addData(self, filePath):
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

    def computeVelocity(self, date1, date2, minPoints=10,
                        dateFormat='%Y-%m-%d'):
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
        None.

        '''
        date, x, y, z, epoch = self.subsetXYZ(date1, date2,
                                              dateFormat=dateFormat)
        if x is np.nan:
            return np.nan, np.nan
        #
        # Uses slope of linear regression as velocity estimate
        vxPS, intercept, rx, px, sigmax = linregress(epoch, x)
        vyPS, intercept, ry, py, sigmay = linregress(epoch, y)
        # Scale from projected to actual coordinates
        return vxPS/self.projLengthScale, vyPS/self.projLengthScale

    def computeVelocityPtToPt(self, date1, date2, minPoints=10,
                              dateFormat='%Y-%m-%d', averagingPeriod=12):
        '''
         Compute velocity for date range differencing point positions

        Parameters
        ----------
        date1 : datetime date
            First date in interval to compute date.
        date2 : datetime date
            Second date in interval to compute date..
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
                                                    minPoints=minPoints)
        dates2, x2, y2, z2, epoch2 = self.subsetXYZ(date2 - tAvg,
                                                    date2 + tAvg,
                                                    minPoints=minPoints)
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
        # Scale from projected to actual coordinates
        return vxPS/self.projLengthScale, vyPS/self.projLengthScale

    def _formatDate(self, date, dateFormat='%Y-%m-%d'):
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

    def subsetXYZ(self, date1, date2, dateFormat='%Y-%m-%d', minPoints=1):
        '''
        Return all x,y, z points in interval [date1, date2]

        Parameters
        ----------
        date1 : TYPE
            DESCRIPTION.
        date2 : TYPE
            DESCRIPTION.
        dateFormat : TYPE, optional
            DESCRIPTION. The default is '%Y-%m-%d'.
        minPoints : TYPE, optional
            Return nan's if # of valid points is < minPoits. The default is 1.

        Returns
        -------
        x, y, z np.array
            x, y, z values in projected coordinates.

        '''
        # Convert to datetime if needed
        date1 = self._formatDate(date1, dateFormat=dateFormat)
        date2 = self._formatDate(date2, dateFormat=dateFormat)
        #
        inRange = np.logical_and(self.date >= date1, self.date <= date2)
        if inRange.sum() < minPoints:
            return np.nan, np.nan, np.nan
        #
        return self.date[inRange], self.x[inRange], self.y[inRange], \
            self.z[inRange], self.epoch[inRange]

    def computeVelocityTimeSeries(self, date1, date2, dT, sampleInterval,
                                  dateFormat='%Y-%m-%d', method='regression',
                                  averagingPeriod=None):
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
        dateFormat : str, optional
            If date1/2 is a str, to datetime format. The default is '%Y-%m-%d'.

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
        vxSeries, vySeries, dateSeries = [], [], []
        #
        # Loop to compute velocities at sample intervale.
        while currentDate + timedelta(hours=dT) < date2:
            if method == 'regression':
                vx, vy = self.computeVelocity(currentDate,
                                              currentDate + timedelta(hours=dT)
                                              )
            elif method == 'point':
                vx, vy = \
                    self.computeVelocityPtToPt(currentDate,
                                               currentDate +
                                               timedelta(hours=dT),
                                               averagingPeriod=averagingPeriod)
            #
            dateSeries.append(currentDate + timedelta(hours=dT/2))
            vxSeries.append(vx)
            vySeries.append(vy)
            currentDate = currentDate + timedelta(hours=sampleInterval)
        return np.array(dateSeries), np.array(vxSeries), np.array(vySeries)
