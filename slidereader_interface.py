import os
import math
from pdb import set_trace
import shutil
import torch
import numpy as np


class SlideReaderCoords(object):
    def __init__(self, slide_name, coord_file):
        # JPEG markers
        self.marker_soi = b'\xff\xd8'
        self.marker_sof0 = b'\xff\xc0'
        self.marker_sos = b'\xff\xda'
        self.tag_nums = {}

        print(slide_name)
        self.slide = open(slide_name, 'rb', 0)
        self.slide_name = slide_name
        self.slide_id = os.path.splitext(os.path.basename(slide_name))[0]
        self.slide.seek(16, os.SEEK_SET)
        self.magicb = self.slide.read(4)
        if self.magicb == b'\xffO\xffQ':
            print('JPEG2000 slide: {}'.format(self.slide_id))
            self.close()
            return

        # Read IFD offsets
        self.level_offsets = self._read_ifd_offsets()
        num_levels = self.level_count()

        self.target_tile_size = 256
        level = 0
        self.level = level
        self._set_level(self.level, op_type='read')
        for x in range(0, self.slide_width, 256):
            for y in range(0, self.slide_height, 256):
                print(x, y)
                coord_file.write('{},{},{},0\n'.format(self.slide_name, x, y))

    def _read_ifd_offsets(self):
        self.slide.seek(0, os.SEEK_SET)
        x = self.slide.read(2)
        if x == b'II':
            self.byte_order = 'little'
        elif x == b'MM':
            self.byte_order = 'big'
        version_num = self.byte_to_int(self.slide.read(2))
        if version_num == 42:
            # Standard TIFF
            self.offset_size = 4
            self.word_size = 2
            self.tag_size = 12
        elif version_num == 43:
            # BigTIFF 64 bit offsets
            self.offset_size = 8
            self.word_size = 8
            self.tag_size = 20
        self.slide.seek(self.offset_size, os.SEEK_SET)
        ifd_offset_bytedata = self.slide.read(self.offset_size)
        ifd_offset = self.byte_to_int(ifd_offset_bytedata)
        level_offsets = []
        while ifd_offset:
            level_offsets.append(ifd_offset)
            self.slide.seek(ifd_offset, os.SEEK_SET)
            no_of_tags = self.byte_to_int(self.slide.read(self.word_size))
            self._read_tagnums(ifd_offset, no_of_tags)
            offset = ifd_offset+ self.word_size+ (no_of_tags* self.tag_size)
            self.slide.seek(offset, os.SEEK_SET)
            ifd_offset_bytedata = self.slide.read(self.offset_size)
            ifd_offset = self.byte_to_int(ifd_offset_bytedata)
        return level_offsets

    def _read_tagnums(self, ifd_offset, no_of_tags):
        self.tag_nums[ifd_offset] = []
        for tag_num in range(no_of_tags):
            tag_offset = ifd_offset+ self.word_size+(tag_num*self.tag_size)
            self.slide.seek(tag_offset, os.SEEK_SET)
            self.tag_nums[ifd_offset].append(
                self.byte_to_int(self.slide.read(2)))

    def _read_tag(self, ifd_offset, tag_name):
        tag_names = {'NewSubfileType': 254, 'ImageWidth': 256,
                    'ImageLength': 257, 'BitsPerSample': 258,
                    'Compression': 259, 'PhotometricInterpretation': 262,
                    'ImageDescription': 270, 'StripOffsets': 273,
                    'SamplesPerPixel': 277, 'RowsPerStrip': 278,
                    'StripByteCounts': 279, 'PlanarConfiguration': 284,
                    'TileWidth': 322, 'TileLength': 323, 'TileOffsets': 324,
                    'TileByteCounts': 325, 'JPEGTables': 347,
                    'YCbCrSubSampling': 530, 'ImageDepth': 32997,
                    'ICCProfile': 34675}
        tag_dtypes = [1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4, 4, 4, 8, 8, 8]
        tag_num = tag_names[tag_name]
        tag_nums = self.tag_nums[ifd_offset]
        tag_idx = tag_nums.index(tag_num)
        tag_offset = ifd_offset+self.word_size+(tag_idx*self.tag_size)
        self.slide.seek(tag_offset, os.SEEK_SET)
        tag_num = self.byte_to_int(self.slide.read(2))
        tag_dtype = self.byte_to_int(self.slide.read(2))
        tag_dtype_bytec = tag_dtypes[tag_dtype-1]
        num_values = self.byte_to_int(self.slide.read(self.offset_size))
        tag_bytedata = self.slide.read(self.offset_size)
        tag_data = self.byte_to_int(tag_bytedata)
        tag_data_bytec = tag_dtype_bytec * num_values
        if tag_data_bytec > self.offset_size:
            self.slide.seek(tag_data, os.SEEK_SET)
            tag_data = self.slide.read(tag_data_bytec)
        # elif num_values > 1:
        #     # TODO check whether this holds well for all tags.
        #     tag_data = tag_bytedata
        return tag_data, num_values, tag_dtype_bytec

    def _set_markers(self, segment_data, level):
        # Read quantization and huffman tables.
        qtht_t = self._read_tag(self.ifd_offset, 'JPEGTables')[0]
        qthttables = qtht_t[2:-2]

        # Change component IDs in start of frame segment.
        sof0marker_idx = segment_data.index(self.marker_sof0)
        sof0segment_len = int.from_bytes(segment_data[sof0marker_idx+2: sof0marker_idx+4], byteorder='big') + 2
        sof0segment_d = segment_data[sof0marker_idx: sof0marker_idx+sof0segment_len]

        sof0segment_ccd = sof0segment_d[:10]+ b'R'+ b'\x11'+ sof0segment_d[12:13]+ b'G'+ b'\x11'+ sof0segment_d[15:16]+ b'B'+ b'\x11'+ sof0segment_d[18:]

        # Change component IDs in start of scan segment.
        self.sosmarker_idx = segment_data.index(self.marker_sos)
        self.sossegment_len = int.from_bytes(segment_data[self.sosmarker_idx+2: self.sosmarker_idx+4],
                                byteorder='big') + 2
        sossegment_d = segment_data[self.sosmarker_idx: self.sosmarker_idx+self.sossegment_len]
        sossegment_ccd = sossegment_d[:5] + b'R' + sossegment_d[6:7] + b'G' + sossegment_d[8:9] + b'B' + sossegment_d[10:] # working
        self.img_g =  self.marker_soi + qthttables + sof0segment_ccd + sossegment_ccd

        return self

    def _set_level(self, level, op_type='read'):
        self.ifd_offset = self.level_offsets[level]
        if 324 in self.tag_nums[self.ifd_offset]:
            data_type = 'Tile'
        else:
            data_type = 'Strip'

        # Data offsets
        data_o, data_count, tag_dtype_bytec = self._read_tag(self.ifd_offset, '{}Offsets'.format(data_type))
        self.segment_offsets = [self.byte_to_int(data_o[n*tag_dtype_bytec: n*tag_dtype_bytec+ tag_dtype_bytec])
                                        for n in range(data_count)]
        # Byte offsets
        tile_bytec, data_count, tag_dtype_bytec = self._read_tag(self.ifd_offset, '{}ByteCounts'.format(data_type))
        self.segment_byte_counts = [self.byte_to_int(tile_bytec[n*tag_dtype_bytec: n*tag_dtype_bytec+ tag_dtype_bytec]) for n in range(data_count)]

        # Read slide dimensions
        self.slide_width = self._read_tag(self.ifd_offset, 'ImageWidth')[0]
        self.slide_height = self._read_tag(self.ifd_offset, 'ImageLength')[0]

        if data_type == 'Tile':
            self.tile_width = self._read_tag(self.ifd_offset, 'TileWidth')[0]
            self.tile_height = self._read_tag(self.ifd_offset, 'TileLength')[0]
            self.tpl = math.ceil(self.slide_width/self.tile_width)
        elif data_type == 'Strip':
            self.rows_per_strip = self._read_tag(self.ifd_offset, 'RowsPerStrip')[0]

        # Read first data to build JPEG markers and segments.
        tile_byte_count = self.segment_byte_counts[0]
        offset_to_tile_data = self.segment_offsets[0]
        self.slide.seek(offset_to_tile_data, os.SEEK_SET)
        segment_data = self.slide.read(tile_byte_count)

        if segment_data[:7] == b'0'*7:
            return
        if op_type == 'read':
            self._set_markers(segment_data, level)

        return self

    def read_write_native_tile(self, tile_n):
        tile_byte_count = self.segment_byte_counts[tile_n]
        offset_to_tile_data = self.segment_offsets[tile_n]
        self.slide.seek(offset_to_tile_data, os.SEEK_SET)
        x = self.slide.read(tile_byte_count)
        image_data_scans = x[self.sosmarker_idx+ self.sossegment_len:]
        img_g = self.img_g + image_data_scans
        x_coord = (tile_n % self.tpl) * self.tile_width
        y_coord = math.floor(tile_n/self.tpl) * self.tile_height
        print(tile_n, x_coord, y_coord, self.tile_width)
        if x_coord > self.slide_width or y_coord > self.slide_height:
            return
        filename = os.path.join(self.tile_dir, '{}_{}_{}.jpg'.format(self.slide_id, x_coord, y_coord))
        target_region_size = 1024
        x_min = x_coord
        y_min = y_coord
        if x_min < 0 or (x_min + (target_region_size*2)) >= self.slide_width or \
            y_min < 0 or (y_min + (target_region_size*2)) >= target_region_size+self.slide_height:
            return
        coord_file.write('{},{},{},0\n'.format(self.slide_name, x_min, y_min))
        # with open(filename, 'wb', 0) as f: f.write(img_g)


    def read_native_tile(self, tile_n):
        tile_byte_count = self.segment_byte_counts[tile_n]
        offset_to_tile_data = self.segment_offsets[tile_n]
        self.slide.seek(offset_to_tile_data, os.SEEK_SET)
        x = self.slide.read(tile_byte_count)
        image_data_scans = x[self.sosmarker_idx+ self.sossegment_len:]
        img_g = self.img_g + image_data_scans
        x_coord = (tile_n % self.tpl) * self.tile_width
        y_coord = math.ceil(tile_n/self.tpl) * self.tile_height
        return img_g

    def read_native_tile_from_coord(self, x_coord, y_coord):
        tile_n = math.ceil(y_coord/self.tile_height) * self.tpl + math.ceil(x_coord/self.tile_width)
        tile_byte_count = self.segment_byte_counts[tile_n]
        offset_to_tile_data = self.segment_offsets[tile_n]
        self.slide.seek(offset_to_tile_data, os.SEEK_SET)
        x = self.slide.read(tile_byte_count)
        image_data_scans = x[self.sosmarker_idx+ self.sossegment_len:]
        img_g = self.img_g + image_data_scans
        x_coord = (tile_n % self.tpl) * self.tile_width
        y_coord = math.ceil(tile_n/self.tpl) * self.tile_height
        return img_g

    def byte_to_int(self, byte_s):
        return int.from_bytes(byte_s, byteorder=self.byte_order, signed=False)

    def dimensions(self, level):
        ifd_offset = self.level_offsets[level]
        return self._read_tag(ifd_offset, 'ImageWidth')[0], self._read_tag(ifd_offset, 'ImageLength')[0]

    def level_count(self):
        return len(self.level_offsets)

    def close(self):
        self.slide.close()
